from sqlalchemy import create_engine, text, inspect
from langchain_openai import ChatOpenAI
from langchain_core.tracers.langchain import LangChainTracer
from langchain_community.callbacks import get_openai_callback
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
import pandas as pd
import json
import re
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# Configure API keys and database path
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DATABASE_PATH = "insurance.db"
CSV_FILE = "data/auto_insurance.csv"
TABLE_NAME = "insurance"
MAX_RESULTS_TO_SHOW = 10
CONTEXT_WINDOW_SIZE = 5  # Store exactly 5 pairs of messages

# Configure LangSmith if needed
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "insurance-chatbot")

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=OPENAI_API_KEY)

# Global chat history storage - list of dictionaries
chat_history = []

# Query context to store the latest data query result and filters
class QueryContext:
    def __init__(self):
        self.sql = None
        self.result = None
        self.columns = None
        self.x_axis = None
        self.y_axis = None
        self.filters = {}  # Store filters like {"State": "California"}

current_query_context = QueryContext()

# Allowed query patterns - to restrict unwanted queries
ALLOWED_QUERY_PATTERNS = [
    r"insurance|policy|claim|premium|coverage|deductible|vehicle|auto|car|driver|accident",
    r"show|compare|find|list|count|average|highest|lowest|most|how many",
    r"chart|graph|plot|visualize|visualization|display",
    r"hello|hi|thanks|thank you|goodbye|bye"
]

# Database setup function
def setup_database():
    """Sets up the SQLite database from CSV."""
    df = pd.read_csv(CSV_FILE)
    engine = create_engine(f"sqlite:///{DATABASE_PATH}")
    df.to_sql(TABLE_NAME, engine, if_exists='replace', index=False)
    logging.debug("Database initialized successfully")
    return engine

# Schema description function
def get_schema_description(engine):
    """Gets schema description for SQL generation."""
    insp = inspect(engine)
    columns = insp.get_columns(TABLE_NAME)
    col_list = [f'"{col["name"]}" ({col["type"]})' for col in columns]
    return f'Table "{TABLE_NAME}" has the following columns:\n' + "\n".join(col_list)

# Check if query is allowed
def is_query_allowed(user_query):
    """Check if the query matches allowed patterns and doesn't contain restricted terms."""
    if not user_query.strip():
        return False

    allowed = any(re.search(pattern, user_query.lower())
                  for pattern in ALLOWED_QUERY_PATTERNS)

    restricted_patterns = [
        r"delete\s+from|drop\s+table|truncate\s+table",
        r"password|credit card|social security"
    ]
    restricted = any(re.search(pattern, user_query.lower())
                     for pattern in restricted_patterns)

    return allowed and not restricted

# Format conversation history for context
def format_conversation_history():
    """Format recent conversation history for context in prompts."""
    global chat_history
    
    # Get the last CONTEXT_WINDOW_SIZE exchanges
    recent_history = chat_history[-CONTEXT_WINDOW_SIZE:] if len(chat_history) > CONTEXT_WINDOW_SIZE else chat_history
    
    # Format messages for context
    formatted_messages = []
    for entry in recent_history:
        formatted_messages.append(f"User: {entry['human_msg'].strip()}")
        formatted_messages.append(f"Assistant: {entry['ai_msg'].strip()}")
    
    return "\n".join(formatted_messages)

# Determine if query is a continuation of previous conversation
def is_query_continuation(user_query, formatted_history):
    """Determine if the current query is a continuation of prior queries."""
    if not formatted_history:
        logging.debug("No history available, treating query as standalone")
        return False
    
    # Check for explicit standalone indicators
    standalone_indicators = ['now show me', 'give me a new', 'all customers', 'every customer']
    if any(indicator in user_query.lower() for indicator in standalone_indicators):
        logging.debug("Standalone indicator detected, treating query as new")
        return False
    
    # Check for vague or follow-up phrases
    follow_up_indicators = ['how many are', 'what about', 'and', 'also', 'now']
    is_vague = any(user_query.lower().startswith(phrase) for phrase in follow_up_indicators)
    
    # Use LLM to analyze semantic continuity
    prompt = f"""
Given the conversation history and current user query, determine if the query is a continuation of the prior context.
- A continuation query refers to the same entity or dataset (e.g., 'how many are married' after 'How many customers are from California?' implies California customers).
- Consider queries with vague references (e.g., 'how many are married') as continuations if they logically extend the prior topic.
- Queries with explicit standalone phrases (e.g., 'now show me', 'all customers') should be treated as new.
- Return 'True' if the query is a continuation, 'False' if it's standalone or shifts topic.

CONVERSATION HISTORY:
{formatted_history}

CURRENT USER QUESTION: {user_query}

Response: True or False
"""
    response = llm.invoke(prompt)
    is_continuation = response.content.strip() == 'True'
    logging.debug(f"Continuation detection: {is_continuation}")
    return is_continuation

# Extract filters from history
def extract_filters_from_history(formatted_history, prior_sql):
    """Extract relevant filters from conversation history and prior SQL."""
    filters = {}
    
    # Parse prior SQL for WHERE conditions
    if prior_sql and 'WHERE' in prior_sql.upper():
        where_clause = prior_sql.split('WHERE')[1].split('GROUP BY')[0] if 'GROUP BY' in prior_sql else prior_sql.split('WHERE')[1]
        conditions = where_clause.split('AND')
        for condition in conditions:
            condition = condition.strip()
            if '=' in condition:
                column, value = condition.split('=')
                column = column.strip().strip('"')
                value = value.strip().strip("'")
                filters[column] = value
    
    # Parse user question from history for additional context
    history_lines = formatted_history.split('\n')
    for line in history_lines:
        if line.startswith('User:'):
            question = line.replace('User:', '').strip().lower()
            # Example: "How many customers are from California?"
            if 'from california' in question:
                filters['State'] = 'California'
            elif 'males' in question or 'male' in question:
                filters['Gender'] = 'M'
            elif 'females' in question or 'female' in question:
                filters['Gender'] = 'F'
    
    # logging.debug(f"Extracted filters: {filters}")
    return filters

# SQL generation function with enhanced context handling
def generate_sql(user_query, schema_description):
    """Generates SQL query using LLM with improved handling of group-by clauses, time-based filtering, and consistent handling of 'top N' queries while ensuring proper context management."""
    # Get formatted conversation history
    formatted_history = format_conversation_history()
    logging.debug(f"Formatted history: {formatted_history}")
    # logging.debug(f"Schema description: {schema_description}")
    
    # Determine if the query is a continuation
    use_context = is_query_continuation(user_query, formatted_history)
    
    # logging.debug(f"Query continuation: {use_context}")
    logging.debug(f"User query: {user_query}")

    # Detect 'top N' queries and extract limit (e.g., "top 5" → N=5)
    top_n_match = re.search(r'\b(top|highest|most)\s*(\d+)?\b', user_query.lower())
    is_top_n = bool(top_n_match)
    top_n_limit = int(top_n_match.group(2)) if top_n_match and top_n_match.group(2) else 3
    
    # Get prior SQL from chat history for filter extraction
    prior_sql = None
    for entry in chat_history[::-1]:
        if entry.get('sql') and not entry['sql'].startswith('Answer from history:'):
            prior_sql = entry['sql']
            break
    
    # Extract filters if continuation
    filters = extract_filters_from_history(formatted_history, prior_sql) if use_context else {}
    
    # Check for time-related phrases in the query
    has_time_reference = re.search(r'\b(year|years|month|months|day|days|recent|latest|last|past|since)\b', user_query.lower())
    
    # Check for dimension/category breakdown requests
    # Look for "by [dimension]" patterns indicating GROUP BY needs
    group_by_match = re.search(r'\b(?:by|per|across|for each|grouped by|break(?:ing)? down by)\s+([a-zA-Z\s]+)\b', user_query.lower())
    needs_group_by = bool(group_by_match)
    
    # Enhanced time filtering guidance based on schema
    time_filtering_guide = """
TIME-BASED FILTERING RULES:
1. For any query with time references (e.g., "last X years", "past X months", "recent"):
   - PREFERRED: Use "Months Since Last Claim" for claim recency (e.g., WHERE "Months Since Last Claim" <= 24 for last 2 years)
   - PREFERRED: Use "Months Since Policy Inception" for policy age/duration (e.g., WHERE "Months Since Policy Inception" <= 24 for policies started in last 2 years)
   - ALTERNATIVE: If "Effective To Date" exists in schema, use it for active policy status (WHERE "Effective To Date" >= current_date)
   - If calculating claim amounts over time periods, prefer "Months Since Last Claim" as the filtering column
   - For ambiguous time references, analyze the intent:
     * "claims in last 2 years" → Use "Months Since Last Claim" <= 24
     * "policies from last 2 years" → Use "Months Since Policy Inception" <= 24
     * "active customers for 2 years" → Use "Months Since Policy Inception" >= 24
2. Examples:
   - Query: "claim amounts for males from Arizona in last 2 years"
     Correct SQL: SELECT SUM("Total Claim Amount") AS total_claim_amount FROM insurance WHERE "Gender" = 'M' AND "State" = 'Arizona' AND "Months Since Last Claim" <= 24;
   - Query: "customers who started policies in past 6 months"
     Correct SQL: SELECT COUNT(*) AS new_customers FROM insurance WHERE "Months Since Policy Inception" <= 6;
"""

    group_by_guide = """
GROUP BY RULES:
1. For queries mentioning "by [dimension]" (e.g., "by gender", "by state", "by age group"):
   - Include the mentioned dimension in the SELECT clause
   - Add a GROUP BY clause for that dimension
   - Example: "average claim amount by gender" → 
     SELECT "Gender", AVG("Total Claim Amount") AS avg_claim_amount FROM insurance GROUP BY "Gender";
2. For multiple dimensions (e.g., "claims by gender and state"):
   - Include all mentioned dimensions in both SELECT and GROUP BY
   - Example: "claims by gender and state" → 
     SELECT "Gender", "State", COUNT(*) as claim_count FROM insurance GROUP BY "Gender", "State";
3. For queries with both filtering conditions AND group by requests:
   - Apply filters with WHERE clause
   - Then apply GROUP BY for the breakdown
   - Example: "average claim amount by gender from Arizona" → 
     SELECT "Gender", AVG("Total Claim Amount") AS avg_claim_amount FROM insurance WHERE "State" = 'Arizona' GROUP BY "Gender";
4. For "by [dimension]" requests:
   - Don't apply additional specific value filters on the grouped dimension
   - Example: For "by gender", don't add WHERE "Gender" = 'M' unless specifically requested
"""

    # Determine if we need the full context handling
    # Either because it's a continuation query or it has special requirements
    need_full_context = use_context or has_time_reference or needs_group_by or is_top_n

    prompt = f"""
DATABASE SCHEMA:
{schema_description}

CONVERSATION HISTORY:
{formatted_history if use_context else 'No relevant history for this query'}

PRIOR FILTERS (apply these if continuation):
{json.dumps(filters) if filters else 'None'}

{time_filtering_guide if has_time_reference else ''}

{group_by_guide if needs_group_by else ''}

CRITICAL INSTRUCTIONS:
1. CONTEXT AWARENESS:
   - If CONVERSATION HISTORY is provided and PRIOR FILTERS exist, include ALL prior filters in the SQL query using AND clauses.
   - For CONTINUATION queries (e.g., 'how many are married' after 'How many customers are from California?'), combine prior filters (e.g., "State" = 'California') with the current query's conditions (e.g., "Marital Status" = 'Married').
   - If the current query is vague or lacks explicit table/column references, assume it applies to the same entity or subset from the prior query (e.g., California customers).
   - If no filters are provided or the query is standalone, ignore prior context and generate SQL based only on the current question.
   - If the query contains standalone indicators (e.g., 'now show me', 'all customers'), treat it as a new query and ignore prior filters.

2. GROUP BY HANDLING:
   - For queries containing "by [dimension]" (e.g., "by gender", "by state", "by age group"):
     * Add the dimension to SELECT clause
     * Use GROUP BY on the dimension
     * Do NOT filter the dimension to a specific value unless explicitly requested
     * Example: "sales by gender" should return data for ALL genders, not just one
   - For phrases like "average X by Y", make Y the GROUP BY dimension
   - IMPORTANT: When grouping by a dimension, don't add arbitrary filters on that dimension

3. TIME-BASED QUERIES:
   - For queries mentioning time periods (e.g., "last 2 years", "past 3 months"):
     * For claim recency/history: Use "Months Since Last Claim" with appropriate comparison
     * For policy age: Use "Months Since Policy Inception" with appropriate comparison
     * Convert years to months (e.g., 2 years = 24 months)
     * Example: "claims in last 2 years" → WHERE "Months Since Last Claim" <= 24
   - For ambiguous time references, analyze the full context to determine which time column is most appropriate.

4. TOP N / HIGHEST QUERIES:
   - For queries containing 'highest', 'top', or 'most' (e.g., 'List customers with highest claim amounts'):
     - Select the top {top_n_limit} rows using ORDER BY <relevant_column> DESC LIMIT {top_n_limit}.
     - If the schema suggests multiple rows per entity (e.g., multiple claims per customer), use aggregation (e.g., MAX or SUM) with GROUP BY to get the total or maximum value per entity.
     - Example: For 'List customers with highest claim amounts', use:
       SELECT "Customer", MAX("Total Claim Amount") AS "Highest Claim Amount"
       FROM insurance
       GROUP BY "Customer"
       ORDER BY "Highest Claim Amount" DESC
       LIMIT {top_n_limit};
   - If the query specifies a number (e.g., 'top 5'), use that number as the LIMIT. Otherwise, default to LIMIT {top_n_limit}.

5. QUERY HANDLING:
   - For CONTINUATION queries: Combine prior filters with the current query's conditions using AND clauses.
   - For NEW queries: Generate SQL based ONLY on the current question, ignoring all previous context.
   - If the answer can be derived directly from the CONVERSATION HISTORY (e.g., arithmetic or rephrasing), return 'Answer from history: [your answer]' instead of generating SQL.
   - If user is asking about the Urban, Suburban, and Rural areas, use the "Location" column to filter for those areas.
   - CRITICAL: If user asking about the examle : "what are the customers married and located in california?" then use the "Marital Status" and "State" columns to filter for those conditions.

6. SQL GENERATION:
   - Use COUNT(*), SUM(), AVG(), etc., for aggregations with clear aliases (e.g., "Number of Married").
   - Use GROUP BY for category distributions or when aggregating per entity (e.g., per customer).
   - Always wrap column names with spaces in double quotes (e.g., "Marital Status").
   - Use precise values: "Gender" ('F', 'M'), "Marital Status" ('Married', 'Single', 'Divorced').
   - Ensure the table name (e.g., 'insurance') is included in the FROM clause.
   - For 'top N' queries, always include ORDER BY and LIMIT unless aggregation makes it unnecessary.
   - Return the SQL query as plain text without any markdown formatting (e.g., no ```sql or ```).
   - If the answer have more than 10 rows, then return the first 10 rows and mention that there are more results available.

7. EXAMPLES:
   - Query: 'List customers with highest claim amounts'
     SELECT "Customer", MAX("Total Claim Amount") AS "Highest Claim Amount"
     FROM insurance
     GROUP BY "Customer"
     ORDER BY "Highest Claim Amount" DESC
     LIMIT {top_n_limit};
   - History: 'How many customers are from California?' → 'SELECT COUNT(*) FROM insurance WHERE "State" = ''California'';'
     Current Query: 'how many are married'
     SELECT COUNT(*) AS "Number of Married"
     FROM insurance
     WHERE "State" = 'California' AND "Marital Status" = 'Married';
   - Query: 'claim amount by males from Arizona in last 2 years'
     SELECT SUM("Total Claim Amount") AS total_claim_amount 
     FROM insurance 
     WHERE "Gender" = 'M' AND "State" = 'Arizona' AND "Months Since Last Claim" <= 24;
   - Query: 'average claim amount by gender from Arizona in last year'
     SELECT "Gender", AVG("Total Claim Amount") AS avg_claim_amount
     FROM insurance
     WHERE "State" = 'Arizona' AND "Months Since Last Claim" <= 12
     GROUP BY "Gender";

CURRENT USER QUESTION: {user_query}

Generate ONLY the appropriate SQL query for SQLite as plain text without any markdown formatting (e.g., no ```sql or ```), unless the answer can be derived directly from history, in which case return 'Answer from history: [your answer]' as plain text.

CRITICAL REMINDERS:
1. TIME-BASED QUERIES:
- For queries about "last X years/months" related to claims, use "Months Since Last Claim" <= X (converting years to months)
- For queries about policy duration/age, use "Months Since Policy Inception"
- Example for "claim data from last 2 years": WHERE "Months Since Last Claim" <= 24

2. GROUP BY REQUESTS:
- For "X by Y" patterns (e.g., "average claim by gender"), include Y in SELECT and GROUP BY clauses
- Example: "average claim amount by gender from Arizona" should return data for ALL genders:
  SELECT "Gender", AVG("Total Claim Amount") AS avg_claim_amount 
  FROM insurance 
  WHERE "State" = 'Arizona' 
  GROUP BY "Gender";
- DO NOT add arbitrary filters on the grouped dimension unless specifically requested by the user

3. CONTEXT HANDLING:
- For follow-up queries, maintain all previous filters and add new ones
- For vague questions like "what about females?", apply context from prior queries
- If user asks for "now show me all customers", reset context and ignore prior filters

"""

    response = llm.invoke(prompt)
    sql = response.content.strip()
    
    # Post-process to remove any markdown backticks
    sql = re.sub(r'```sql\n|```', '', sql).strip()
    
    print(f"Generated SQL: {sql}")
    logging.debug(f"Generated SQL: {sql}")
    return sql

# SQL execution function
def run_sql(sql_query, engine):
    """Execute SQL and format results."""
    with engine.connect() as conn:
        result = conn.execute(text(sql_query))
        print(f"SQL executed22: {sql_query}")
        rows = result.fetchall()
        columns = result.keys()

        if len(rows) == 1 and len(columns) == 1:
            return str(rows[0][0])
        elif len(rows) == 1:
            return dict(zip(columns, rows[0]))
        elif len(rows) > MAX_RESULTS_TO_SHOW:
            preview = [dict(zip(columns, row))
                       for row in rows[:MAX_RESULTS_TO_SHOW]]
            remaining = len(rows) - MAX_RESULTS_TO_SHOW
            return {"data": preview, "more": remaining}
        else:
            return [dict(zip(columns, row)) for row in rows]

# Analyze data structure to determine if it's suitable for visualization
def analyze_data_structure(result_data):
    """Analyze data structure to determine if it's suitable for visualization."""
    # Handle empty data
    if not result_data:
        return False, "No data available for visualization"
    
    # Handle scalar value
    if isinstance(result_data, str) and result_data.isdigit():
        return False, "Single numeric value is not suitable for visualization"
        
    # Handle dictionaries with "data" key
    if isinstance(result_data, dict) and "data" in result_data:
        if not result_data["data"]:
            return False, "Empty result set"
        data_to_analyze = result_data["data"]
    elif isinstance(result_data, list):
        if not result_data:
            return False, "Empty result set"
        data_to_analyze = result_data
    else:
        return False, "Data structure not suitable for visualization"
    
    # Check if there's at least one record
    if len(data_to_analyze) < 1:
        return False, "Not enough data points for visualization"
        
    # Check for numeric values to plot
    has_numeric = False
    for item in data_to_analyze:
        for key, value in item.items():
            if isinstance(value, (int, float)) or (isinstance(value, str) and value.replace('.', '').isdigit()):
                has_numeric = True
                break
        if has_numeric:
            break
            
    if not has_numeric:
        return False, "No numeric values found for visualization"
        
    # Check for categorical dimension
    if len(data_to_analyze) <= 1:
        return False, "Not enough data points for comparison"
        
    return True, "Data is suitable for visualization"

# Updated identify_axes function for better axis selection
def identify_axes(data):
    """Identify the most appropriate X and Y axes for visualization."""
    if not data or (isinstance(data, list) and len(data) == 0):
        return None, None
        
    if isinstance(data, dict) and "data" in data:
        items = data["data"]
    elif isinstance(data, list):
        items = data
    else:
        return None, None
        
    if len(items) == 0:
        return None, None
        
    # Get all keys from the first item
    keys = list(items[0].keys())
    
    # Categorize keys by data type
    numeric_keys = []
    categorical_keys = []
    
    for key in keys:
        # Check if all values for this key are numeric
        all_numeric = True
        values = set()
        
        for item in items:
            value = item.get(key)
            values.add(value)
            
            if not isinstance(value, (int, float)) and not (isinstance(value, str) and value.replace('.', '').isdigit()):
                all_numeric = False
                
        # If fewer than 15 unique values, consider it categorical
        if len(values) < 15 and not all_numeric:
            categorical_keys.append(key)
        elif all_numeric:
            numeric_keys.append(key)
        else:
            # Text fields with many unique values aren't useful for visualization
            pass
            
    # Ideal: one categorical for x-axis, one numeric for y-axis
    if categorical_keys and numeric_keys:
        # Prefer "Income", "State", "Gender", "Marital Status" as x-axis if available
        preferred_x = ["Income", "State", "Gender", "Marital Status", "Vehicle Class", "Coverage"]
        for preferred in preferred_x:
            if preferred in categorical_keys:
                return preferred, numeric_keys[0]
        return categorical_keys[0], numeric_keys[0]
    
    # Only categorical: use first as x, COUNT(*) as y
    elif categorical_keys:
        count_columns = [k for k in keys if 'count' in k.lower() or 'total' in k.lower() or 'avg' in k.lower()]
        if count_columns:
            return categorical_keys[0], count_columns[0]
        else:
            return categorical_keys[0], "COUNT(*)"
    
    # Only numeric: use first as x, second as y
    elif len(numeric_keys) >= 2:
        # Prefer keys like "Income" for x-axis if available
        if "Income" in numeric_keys:
            return "Income", [k for k in numeric_keys if k != "Income"][0]
        return numeric_keys[0], numeric_keys[1]
    
    # One numeric: use index as x, numeric as y
    elif len(numeric_keys) == 1:
        return "Index", numeric_keys[0]
    
    # No good candidates
    return None, None

# Determine if visualization is appropriate
def should_visualize(user_query, result_data):
    """Determine if visualization is appropriate based on query and data."""
    global chat_history
    
    # Check if user explicitly requested visualization
    viz_terms = ["chart", "graph", "plot", "visualize", "visualization", "display"]
    explicit_request = any(term in user_query.lower() for term in viz_terms)
    
    # Analyze data structure
    suitable, reason = analyze_data_structure(result_data)
    
    if not chat_history:
        return False, "No data available for visualization"
    elif explicit_request and not suitable:
        return False, f"I can't create a visualization because: {reason}"
    elif explicit_request and suitable:
        return True, "User requested visualization and data is suitable"
    elif suitable:
        # Check if the data has a good structure for visualization
        x_axis, y_axis = identify_axes(result_data)
        if x_axis and y_axis:
            
            return True, "Data structure is suitable for visualization"
        else:
            return False, "Data structure doesn't have clear axes for visualization"
    else:
        return False, "Data not suitable for visualization"

# Updated prepare_visualization_data for correct data formatting
def prepare_visualization_data(result_data):
    """Format result data for chart visualization with proper axes."""
    if isinstance(result_data, dict) and "data" in result_data:
        chart_data = result_data["data"]
    elif isinstance(result_data, list):
        chart_data = result_data
    else:
        return None, None, None
    
    if not chart_data:
        return None, None, None
    
    if len(chart_data) > 15:
        chart_data = chart_data[:15]  # Limit to 15 data points for readability
    
    # Identify best axes for this data
    x_axis, y_axis = identify_axes(chart_data)
    
    if not x_axis or not y_axis:
        return None, None, None
    
    # For COUNT(*) we need to add the count manually
    if y_axis == "COUNT(*)":
        count_field = "Count"
        for item in chart_data:
            item[count_field] = 1
        y_axis = count_field
    
    # Format data for frontend
    formatted_data = []
    for item in chart_data:
        x_value = item.get(x_axis)
        y_value = item.get(y_axis)
        if x_value is not None and y_value is not None:
            formatted_data.append({
                "x": str(x_value),  # Convert to string for consistency
                "y": float(y_value) if isinstance(y_value, (int, float)) else float(y_value.replace(',', '')) if isinstance(y_value, str) and y_value.replace(',', '').replace('.', '').isdigit() else 0
            })
    
    return formatted_data, x_axis, y_axis

# Updated determine_chart_type for better chart selection
def determine_chart_type(user_query, chart_data, x_axis, y_axis):
    """Determine the best chart type based on data characteristics."""
    # Check if user explicitly requested a chart type
    query = user_query.lower()
    
    if re.search(r'\b(bar|column)\s*(chart|graph|plot|viz)', query):
        return "bar"
    elif re.search(r'\b(pie|circle|donut)\s*(chart|graph|plot|viz)', query):
        return "pie"
    elif re.search(r'\b(line|trend|time)\s*(chart|graph|plot|viz)', query):
        return "line"
    elif re.search(r'\b(scatter|point|dots)\s*(chart|graph|plot|viz)', query):
        return "scatter"
    
    # Auto-determine based on data characteristics
    if not chart_data or not x_axis or not y_axis:
        return "bar"  # Default
    
    # Count unique x values
    unique_x_values = set(item["x"] for item in chart_data)
    
    # Check if x-axis might be time-related
    time_related = any(term in x_axis.lower() for term in ["date", "time", "year", "month", "day"])
    
    # Check if y-axis is a percentage or proportion
    is_percentage = "percentage" in y_axis.lower() or all(float(item["y"]) <= 100 for item in chart_data)
    
    # Rules for chart type selection
    if time_related:
        return "line"
    elif len(unique_x_values) <= 7 and is_percentage:
        return "pie"
    elif len(unique_x_values) <= 7 and len(chart_data) <= 7:
        return "pie"
    elif len(unique_x_values) > 10 and not is_percentage:
        return "scatter"
    else:
        return "bar"
    
def extract_limit_from_query(user_query):
    """
    Extract numeric limit from user query like "top 10", "first 5", etc.
    Returns default value of 6 if no limit is found.
    """
    # Look for patterns like "top N", "first N", "best N", "N most", etc.
    patterns = [
        r"top\s+(\d+)",
        r"first\s+(\d+)",
        r"best\s+(\d+)",
        r"(\d+)\s+most",
        r"(\d+)\s+highest",
        r"(\d+)\s+lowest",
        r"(\d+)\s+best",
        r"(\d+)\s+top",
        r"show\s+(\d+)",
        r"list\s+(\d+)",
        r"give\s+(\d+)",
        r"display\s+(\d+)"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, user_query.lower())
        if match:
            try:
                return int(match.group(1))
            except (ValueError, IndexError):
                pass
    
    # Default limit if no pattern is found
    return 6

def generate_user_friendly_response(user_query, result_data, sql_query, has_visualization=False):
    """
    Generates a conversational, user-friendly response to data queries.
    Dynamically handles result limits based on user query.
    """
    # Extract limit from user query
    dynamic_limit = extract_limit_from_query(user_query)
    
    # Format the result data for the prompt
    if isinstance(result_data, dict) and "data" in result_data:
        data = result_data["data"]
        result_summary = json.dumps(data[:dynamic_limit], indent=2) if len(data) > dynamic_limit else json.dumps(data, indent=2)
        more_count = max(0, len(data) - dynamic_limit) if len(data) > dynamic_limit else 0
    elif isinstance(result_data, list):
        result_summary = json.dumps(result_data[:dynamic_limit], indent=2) if len(result_data) > dynamic_limit else json.dumps(result_data, indent=2)
        more_count = max(0, len(result_data) - dynamic_limit) if len(result_data) > dynamic_limit else 0
    else:
        result_summary = str(result_data)
        more_count = 0

    prompt = f"""
Create a conversational, friendly response to the user's question using the SQL query results.

User question: {user_query}
SQL query executed: {sql_query}
SQL result: {result_summary}
Additional results not shown: {more_count}
Has visualization: {has_visualization}

Guidelines:
1. Be conversational and personable, like a helpful assistant
2. Address the user's question directly and completely
3. Highlight the most important insights from the data
4. Use natural language (not technical database terms)
5. Keep your answer concise but complete
6. Avoid mentioning SQL or database terminology
7. For many results, summarize instead of listing them all
8. When describing numerical results, be precise with numbers
9. If there's a visualization, mention it briefly but don't focus on it

Be friendly and helpful, but get straight to the point.
"""

    response = llm.invoke(prompt)
    return response.content.strip()

# Generate user-friendly response
# def generate_user_friendly_response(user_query, result_data, sql_query, has_visualization):
#     """Generates a conversational, user-friendly response to data queries."""
#     # Format the result data for the prompt
#     if isinstance(result_data, dict) and "data" in result_data:
#         data = result_data["data"]
#         result_summary = json.dumps(data[:6], indent=2) if len(data) > 6 else json.dumps(data, indent=2)
#         more_count = result_data.get("more", 0)
#     elif isinstance(result_data, list):
#         result_summary = json.dumps(result_data[:6], indent=2) if len(result_data) > 6 else json.dumps(result_data, indent=2)
#         more_count = max(0, len(result_data) - 6) if len(result_data) > 6 else 0
#     else:
#         result_summary = str(result_data)
#         more_count = 0

#     prompt = f"""
# Create a conversational, friendly response to the user's question using the SQL query results.

# User question: {user_query}
# SQL query executed: {sql_query}
# SQL result: {result_summary}
# Additional results not shown: {more_count}


# Guidelines:
# 1. Be conversational and personable, like a helpful assistant
# 2. Address the user's question directly and completely
# 3. Highlight the most important insights from the data
# 4. Use natural language (not technical database terms)
# 5. Keep your answer concise but complete
# 6. Avoid mentioning SQL or database terminology
# 8. For many results, summarize instead of listing them all
# 9. When describing numerical results, be precise with numbers

# Be friendly and helpful, but get straight to the point.
# """

#     response = llm.invoke(prompt)
#     return response.content.strip()

# Main database query function with direct visualization handling
def ask_question(user_query, schema):
    """Processes database queries with direct visualization handling."""
    global current_query_context
    try:
        engine = setup_database()

        with get_openai_callback() as cb:
            # Generate SQL
            sql_query = generate_sql(user_query, schema)
            print(f"Generated SQL: {sql_query}")
            # logging.debug(f"SQL query for {user_query}: {sql_query}")
            
            # Check if the response is from history
            if sql_query.startswith("Answer from history:"):
                response_text = sql_query.replace("Answer from history:", "").strip()
                result_data = None
                visualization_data = None
                # Clear query context for non-data queries
                current_query_context = QueryContext()
            else:
                result_data = run_sql(sql_query, engine)
                print(f"Query result: {result_data}")
                
                # logging.debug(f"Query result: {result_data}")
                
                # Update query context
                current_query_context.sql = sql_query
                current_query_context.result = result_data
                current_query_context.columns = list(result_data[0].keys()) if isinstance(result_data, list) else None
                # Extract filters from SQL
                current_query_context.filters = extract_filters_from_history(format_conversation_history(), sql_query)
                
                # Check if visualization is appropriate
                should_viz, viz_reason = should_visualize(user_query, result_data)
                # logging.debug(f"Visualization decision: {should_viz}, Reason: {viz_reason}")

                # Handle visualization if appropriate
                visualization_data = None
                if should_viz:
                    chart_data, x_axis, y_axis = prepare_visualization_data(result_data)
                    # logging.debug(f"Visualization axes: x={x_axis}, y={y_axis}")
                    
                    if chart_data and x_axis and y_axis:
                        chart_type = determine_chart_type(user_query, chart_data, x_axis, y_axis)
                        # logging.debug(f"Chart type: {chart_type}")
                        
                        # Prepare data specifically for the chosen chart type
                        if chart_type == "pie" and len(chart_data) > 7:
                            chart_data = sorted(chart_data, key=lambda x: float(x["y"]), reverse=True)[:7]
                        
                        visualization_data = {
                            "type": chart_type,
                            "data": chart_data,
                            "x_axis": x_axis,
                            "y_axis": y_axis,
                            "x_label": x_axis.replace("_", " ").title(),
                            "y_label": y_axis.replace("_", " ").title()
                        }
                        # Update context with visualization axes
                        current_query_context.x_axis = x_axis
                        current_query_context.y_axis = y_axis
                
                # Generate user-friendly response
                
                print(f"Response data11: {result_data}")
                
                response_text = generate_user_friendly_response(user_query, result_data, sql_query, should_viz)
                
                logging.debug(f"Response text: {response_text}")
                
                # If user explicitly asked for visualization but it's not possible, add explanation
                if any(term in user_query.lower() for term in ["chart", "graph", "visual"]):
                    if not should_viz:
                        response_text += f" {viz_reason}"
            
            # Add to global chat history
            global chat_history
            chat_history.append({
                "human_msg": user_query,
                "ai_msg": response_text,
                "sql": sql_query,
                "visualization": visualization_data
            })
            
            return {
                "text": response_text,
                "visualization": visualization_data,
                "sql": sql_query
            }
    except Exception as e:
        logging.error(f"Error in ask_question: {str(e)}")
        return {"text": f"I'm focused on helping with auto insurance question. Is there something about your auto policy, coverage, or insurance claims I can help with instead?", "visualization": None, "sql": None}

# Improved general query function for insurance definitions and concepts
def ask_general_question(query, bot_name="Aira"):
    """Handles general non-database queries with natural responses, focusing on insurance concepts."""
    try:
        # Declare global variables at the beginning of the function
        global chat_history
        
        # Get conversation history for context
        formatted_history = format_conversation_history()
        
        # Check for greetings or capability questions
        query_lower = query.lower()
        
        # Handle name questions
        if any(phrase in query_lower for phrase in ["what's your name", "what is your name", 
                                                    "who are you", "your name", "what should i call you"]):
            name_response = f"My name is {bot_name}! I'm your auto insurance assistant, ready to help you analyze insurance data and answer any questions you might have."
            
            # Add to global chat history
            chat_history.append({
                "human_msg": query,
                "ai_msg": name_response,
                "sql": None,
                "visualization": None
            })
            
            return {"text": name_response, "visualization": None, "sql": None}
        
        # Handle capability questions specifically
        if any(phrase in query_lower for phrase in ["what can you do", "your capabilities", "what are you able to",
                                                   "help me with", "what do you know", "how can you help",
                                                   "what can you help with", "what are your abilities"]):
            capability_response = f"""I'm {bot_name}, here to help you gain valuable insights from your auto insurance claims database. You can ask about claim amounts, customer demographics, policy types, vehicle details, coverage patterns, and sales channel trends. I deliver clear, data-driven responses based on your records."""
            
            # Add to global chat history
            chat_history.append({
                "human_msg": query,
                "ai_msg": capability_response,
                "sql": None,
                "visualization": None
            })
            
            return {"text": capability_response, "visualization": None, "sql": None}
        
        # Detect if query is a greeting
        is_greeting = any(greeting in query_lower for greeting in ["hello", "hi ", "hey", "good morning", "good afternoon",
                                                                  "good evening", "howdy", "what's up", "greetings"])
        
        if is_greeting:
            # Create a prompt to generate a dynamic greeting response
            greeting_prompt = f"""
You are {bot_name}, a friendly auto insurance assistant responding to a greeting.

USER GREETING: {query}

Guidelines for your response:
- Introduce yourself as {bot_name}
- Respond with a personalized greeting that matches the user's tone and time-of-day reference (if any)
- If the user says "Good morning", respond with an enhanced morning greeting
- If the user says "Good afternoon", respond with an enhanced afternoon greeting
- If the user says "Good evening", respond with an enhanced evening greeting
- Keep it brief and warm (1-2 sentences)
- Mention you're an auto insurance assistant ready to help
- Include that you can provide insurance information and data insights
- Be conversational and natural
- Avoid sounding robotic

Example responses:
- For "Good morning": "A wonderful morning to you! I'm {bot_name}, your auto insurance assistant ready to provide insights and answer your insurance questions."
- For "Hi there": "Hi there! I'm {bot_name}, your auto insurance assistant, ready to dive into insurance data and answer any questions you might have."

Generate ONLY the greeting response, nothing else.
"""
            # Get dynamic greeting from LLM
            greeting_response = llm.invoke(greeting_prompt).content.strip()
            
            # Add to global chat history
            chat_history.append({
                "human_msg": query,
                "ai_msg": greeting_response,
                "sql": None,
                "visualization": None
            })
            
            return {"text": greeting_response, "visualization": None, "sql": None}
        
        # Handle other general questions
        prompt = f"""
You are {bot_name}, a friendly auto insurance assistant having a conversation. You specialize in explaining insurance concepts and providing helpful information.

CONVERSATION HISTORY:
{formatted_history}

CURRENT QUESTION:
{query}

Guidelines for your response:
- Always refer to yourself as {bot_name} if you need to mention your name
- Be warm and conversational, like you're chatting with a friend
- If the question is about an insurance concept or definition (like "What is a deductible?", "How does comprehensive coverage work?"), provide a CLEAR and CONCISE explanation
- For general insurance questions, focus on being INFORMATIVE while using simple language
- For greetings or social messages, be BRIEF and FRIENDLY
- Keep your answers focused on auto insurance topics when relevant
- Avoid insurance jargon when possible, or explain it immediately if you use it
- Be concise but helpful (2-4 sentences is ideal)
- Sound human and conversational, not like a technical manual
"""
        response = llm.invoke(prompt)
        
        # Add to global chat history
        chat_history.append({
            "human_msg": query,
            "ai_msg": response.content,
            "sql": None,
            "visualization": None
        })
        
        return {"text": response.content, "visualization": None, "sql": None}
    except Exception as e:
        logging.error(f"Error in ask_general_question: {str(e)}")
        return {"text": "I'm sorry, I couldn't process that. Could you try again?", "visualization": None, "sql": None}
    
# Query classification with improved context-awareness
# def classify_query(user_query, schema):
#     """Uses LLM to classify the intent of the query with context-aware logic."""
#     try:
#         # Get current conversation history
#         formatted_history = format_conversation_history()
        
#         classification_prompt = f"""
# You are an intent classifier for an auto insurance chatbot.

# CONVERSATION HISTORY:
# {formatted_history}

# USER QUERY TO CLASSIFY:
# "{user_query}"

# Your job is to classify the query into ONE of these categories:

# 1. **data_query** – The user is asking about:
#    - Customer data, insurance policies, claims, premiums, payments from the database
#    - Statistics/aggregations about the data (counts, averages, etc.)
#    - Follow-up questions that logically extend previous data queries (e.g., asking for additional filters or details on the same topic)
#    - Examples: "How many customers are married?", "What about California customers?", "Show premiums for females"

# 2. **general_chat** – The user is asking about:
#    - Asking regarding capabilities.
#    - Greetings, thanks, closings (e.g., "Hello", "Thanks", "Goodbye")

# 3. **out_of_scope** – Completely unrelated to auto insurance:
#    - CRITICAL: Asking about the defination like "what is auto insurance?"
#    - Weather, sports, news, other types of insurance, etc.
#    - Asking about the ai defination. e.g "what is auto insurance?"
#    - Asking about the insurance defination and any other defination.
#    - Asking for definitions or explanations of unrelated topics
#    - CRITICAL: Examples: "What's the weather?", "Who won the World Cup?", "Tell me about health insurance", "What is auto insurance?", "What is insurance?"

# 4. **visualization_only** – Request to visualize previous data:
#    - Clear request for a visualization with no new data criteria
#    - Examples: "Make a pie chart", "Show that as a bar graph", "Visualize the data"

# 5. **visualization_with_criteria** – Visualization with new filters:
#    - Examples: "Bar chart for female customers", "Pie chart of complaints by region"

# CLASSIFICATION RULES:
# 1. FOLLOW-UP DETECTION: Queries that logically extend the previous topic (same entities, attributes, or data focus) are data_query, even without explicit references. Use semantic similarity to determine continuation.
# 2. NEW TOPIC DETECTION: Queries with phrases like "now show me", "give me a new", or a clear topic shift (e.g., from data to definitions) are treated as standalone.
# 3. DATA QUESTIONS: Any query that could be answered from the database schema is data_query.
# 4. GREETINGS: Simple greetings, thanks.
# 5. Don't answer insurance related questions that are not provided in the database schema.

# Respond with ONLY one of: "data_query", "general_chat", "out_of_scope", "visualization_only", or "visualization_with_criteria"
# """

#         response = llm.invoke(classification_prompt)
#         classification = response.content.strip().lower()
#         logging.debug(f"Query classification: {classification}")

#         valid_classes = ["data_query", "general_chat", "out_of_scope",
#                          "visualization_only", "visualization_with_criteria"]
        
#         if classification in valid_classes:
#             return classification
#         else:
#             return "data_query"  # Default to data_query when uncertain
#     except Exception as e:
#         logging.error(f"Classification error: {str(e)}")
#         return "general_chat"


def classify_query(user_query, schema):
    """Uses LLM to classify the intent of the query with context-aware logic."""
    try:
        # Get current conversation history
        formatted_history = format_conversation_history()
        
        classification_prompt = f"""
You are an intent classifier for an auto insurance chatbot.

CONVERSATION HISTORY:
{formatted_history}

USER QUERY TO CLASSIFY:
"{user_query}"

Your job is to classify the query into ONE of these categories:

1. **data_query** – The user is asking about:
   - Customer data, insurance policies, claims, premiums, payments from the database
   - Statistics/aggregations about the data (counts, averages, etc.)
   - Follow-up questions that logically extend previous data queries (e.g., asking for additional filters or details on the same topic)
   - Examples: "How many customers are married?", "What about California customers?", "Show premiums for females"

2. **general_chat** – The user is asking about:
   - Asking regarding capabilities.
   - Greetings, thanks, closings (e.g., "Hello", "Thanks", "Goodbye")

3. **out_of_scope** – Completely unrelated to auto insurance:
   - CRITICAL: Asking about definitions like "what is auto insurance?" or "what is insurance?"
   - Weather, sports, news, other types of insurance, etc.
   - Asking for definitions or explanations of insurance concepts or any other topics
   - CRITICAL: Examples: "What's the weather?", "Who won the World Cup?", "Tell me about health insurance", "What is auto insurance?", "What is insurance?", "Define auto insurance", "Can you explain what insurance is?"

4. **visualization_only** – Request to visualize previous data:
   - Clear request for a visualization with no new data criteria
   - Examples: "Make a pie chart", "Show that as a bar graph", "Visualize the data"

5. **visualization_with_criteria** – Visualization with new filters:
   - Examples: "Bar chart for female customers", "Pie chart of complaints by region"

CLASSIFICATION RULES:
1. FOLLOW-UP DETECTION: Queries that logically extend the previous topic (same entities, attributes, or data focus) are data_query, even without explicit references. Use semantic similarity to determine continuation.
2. NEW TOPIC DETECTION: Queries with phrases like "now show me", "give me a new", or a clear topic shift (e.g., from data to definitions) are treated as standalone.
3. DATA QUESTIONS: Any query that could be answered from the database schema is data_query.
4. GREETINGS: Simple greetings, thanks.
5. DEFINITION QUESTIONS: Any query starting with "what is", "define", "explain what", or similar phrases followed by insurance-related terms should be classified as out_of_scope. Examples include "what is auto insurance?", "what is insurance?", "explain what car insurance is".

Respond with ONLY one of: "data_query", "general_chat", "out_of_scope", "visualization_only", or "visualization_with_criteria"
"""

        response = llm.invoke(classification_prompt)
        classification = response.content.strip().lower()
        logging.debug(f"Query classification: {classification}")

        valid_classes = ["data_query", "general_chat", "out_of_scope",
                         "visualization_only", "visualization_with_criteria"]
        
        if classification in valid_classes:
            return classification
        else:
            return "data_query"  # Default to data_query when uncertain
    except Exception as e:
        logging.error(f"Classification error: {str(e)}")
        return "general_chat"


# Updated handle_visualization_only for correct visualization
def handle_visualization_only(user_query):
    """Handles requests to visualize the most recent data query result."""
    global current_query_context
    
    # Check if there is a valid query context with data
    if not current_query_context.result:
        response_text = "I don't have any recent data to visualize. Please provide a data query first."
        logging.debug("No data available for visualization")
        chat_history.append({
            "human_msg": user_query,
            "ai_msg": response_text,
            "sql": None,
            "visualization": None
        })
        return {
            "text": response_text,
            "visualization": None,
            "sql": None
        }

    # Use the current query context's result
    result_data = current_query_context.result
    
    # Check if data is visualizable and prepare it
    chart_data, x_axis, y_axis = prepare_visualization_data(result_data)
    logging.debug(f"Preparing visualization: x_axis={x_axis}, y_axis={y_axis}")
    
    if not chart_data or not x_axis or not y_axis:
        response_text = "The recent data doesn't have a structure I can visualize effectively. Could you try a different query that returns data with categories and numbers?"
        logging.debug("Data not suitable for visualization")
        chat_history.append({
            "human_msg": user_query,
            "ai_msg": response_text,
            "sql": None,
            "visualization": None
        })
        return {
            "text": response_text,
            "visualization": None,
            "sql": None
        }

    # Determine the best chart type based on user request and data
    chart_type = determine_chart_type(user_query, chart_data, x_axis, y_axis)
    logging.debug(f"Selected chart type: {chart_type}")
    
    # Prepare visualization data
    visualization_data = {
        "type": chart_type,
        "data": chart_data,
        "x_axis": x_axis,
        "y_axis": y_axis,
        "x_label": x_axis.replace("_", " ").title(),
        "y_label": y_axis.replace("_", " ").title()
    }

    # Generate response
    if chart_type == "bar":
        response_text = f"Here's a bar chart showing {y_axis.replace('_', ' ').title()} across different {x_axis.replace('_', ' ').title()} values."
    elif chart_type == "pie":
        response_text = f"Here's a pie chart showing the distribution of {y_axis.replace('_', ' ').title()} across {x_axis.replace('_', ' ').title()} categories."
    elif chart_type == "line":
        response_text = f"Here's a line chart showing how {y_axis.replace('_', ' ').title()} changes across {x_axis.replace('_', ' ').title()} values."
    else:
        response_text = f"Here's a {chart_type} chart visualization of the recent data."

    # Add to global chat history
    chat_history.append({
        "human_msg": user_query,
        "ai_msg": response_text,
        "sql": current_query_context.sql,
        "visualization": visualization_data
    })

    return {
        "text": response_text,
        "visualization": visualization_data,
        "sql": None
    }

# Out of scope handler
def handle_out_of_scope(query):
    """Handles questions that are not relevant to insurance."""
    response_text = "I'm focused on helping with auto insurance questions. Is there something about your auto policy, coverage, or insurance claims I can help with instead?"
    
    # Add to global chat history
    global chat_history
    chat_history.append({
        "human_msg": query,
        "ai_msg": response_text,
        "sql": None,
        "visualization": None
    })
    
    return {"text": response_text, "visualization": None, "sql": None}

# Main query handler
def handle_query(user_query, schema):
    """Routes the query to the correct agent based on classification."""
    # print(f"database: {schema}")
    query_type = classify_query(user_query, schema)
    logging.debug(f"Query type detected: {query_type}")

    if query_type == "visualization_only":
        logging.debug(f"Visualization only detected: {user_query}")
        return handle_visualization_only(user_query)
    elif query_type == "visualization_with_criteria":
        logging.debug(f"Visualization with criteria detected: {user_query}")
        return ask_question(user_query, schema)
    elif query_type == "data_query":
        logging.debug(f"Data query detected: {user_query}")
        return ask_question(user_query, schema)
    elif query_type == "general_chat":
        logging.debug(f"General chat detected: {user_query}")
        return ask_general_question(user_query)
    else:
        return handle_out_of_scope(user_query)

# Flask route for handling chat messages
@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message", "")

    engine = setup_database()
    schema = get_schema_description(engine)
    logging.info("Database initialized successfully")

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    # Check conversation length to apply context window
    global chat_history
    
    # If reached CONTEXT_WINDOW_SIZE pairs, keep only the most recent
    if len(chat_history) > CONTEXT_WINDOW_SIZE:
        chat_history = chat_history[-CONTEXT_WINDOW_SIZE:]

    # Process the query
    response = handle_query(user_message, schema)

    # Return response with visualization if available
    return jsonify({
        "response": response["text"],
        "visualization": response["visualization"],
        "sql": response["sql"]
    })

# Health check endpoint
@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"})

# Reset conversation history endpoint
@app.route("/api/reset", methods=["POST"])
def reset_conversation():
    global chat_history, current_query_context
    chat_history = []
    current_query_context = QueryContext()
    logging.info("Conversation history and query context reset")
    return jsonify({"status": "conversation reset"})

if __name__ == "__main__":
    app.run(debug=True, host="localhost", port=5000)