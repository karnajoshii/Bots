from sqlalchemy import create_engine, text, inspect
from langchain_openai import ChatOpenAI
from langchain_community.callbacks import get_openai_callback
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
import pandas as pd
import json
import re
import logging
import mysql.connector
from mysql.connector import Error
import uuid
import datetime

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

# MySQL configuration
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")

# Configure LangSmith if needed
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "insurance-chatbot")

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=OPENAI_API_KEY)

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

# Database setup function for SQLite
def setup_database():
    """Sets up the SQLite database from CSV."""
    df = pd.read_csv(CSV_FILE)
    engine = create_engine(f"sqlite:///{DATABASE_PATH}")
    df.to_sql(TABLE_NAME, engine, if_exists='replace', index=False)
    logging.debug("Database initialized successfully")
    return engine

# MySQL connection utility
def create_db_connection():
    """Create a connection to the MySQL database."""
    try:
        connection = mysql.connector.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME
        )
        return connection
    except Error as e:
        logging.error(f"Database connection error: {e}")
        return None

# Execute MySQL query
def execute_query(connection, query, params=None, fetch=True):
    """Execute a MySQL query with optional parameters."""
    cursor = connection.cursor(dictionary=True)
    try:
        cursor.execute(query, params)
        if fetch:
            result = cursor.fetchall()
            connection.commit()
            return result
        else:
            connection.commit()
            return cursor
    except Error as e:
        logging.error(f"Error executing query: {e}")
        connection.rollback()
        raise e
    finally:
        cursor.close()

# Get client IP address
def get_client_ip():
    """Get the client's IP address from request."""
    if request.environ.get('HTTP_X_FORWARDED_FOR'):
        return request.environ['HTTP_X_FORWARDED_FOR'].split(',')[0].strip()
    elif request.environ.get('HTTP_X_REAL_IP'):
        return request.environ['HTTP_X_REAL_IP']
    else:
        return request.remote_addr

# Get or create chat session
def get_or_create_chat_session_by_client_id(client_id):
    """Get existing active chat session for client ID or create a new one."""
    connection = create_db_connection()
    if connection is None:
        raise Exception("Database connection failed")
    
    try:
        query = """
            SELECT id FROM chat_sessions 
            WHERE client_id = %s AND deleted = FALSE 
            ORDER BY created_at DESC LIMIT 1
        """
        results = execute_query(connection, query, (client_id,), fetch=True)
        
        if results and len(results) > 0:
            return results[0]['id']
        else:
            new_chat_id = str(uuid.uuid4())
            insert_query = """
                INSERT INTO chat_sessions (id, client_id, created_at, deleted) 
                VALUES (%s, %s, NOW(), FALSE)
            """
            execute_query(connection, insert_query, (new_chat_id, client_id), fetch=False)
            return new_chat_id
    finally:
        if connection and connection.is_connected():
            connection.close()

# Save chat message
def save_chat_message(chat_id, role, message, sql_query=None, visualization=None):
    """Save a chat message to the database."""
    connection = create_db_connection()
    if connection is None:
        logging.error("Failed to save chat message: No database connection")
        return False
    
    try:
        # Check for duplicate message within 5 seconds
        check_query = """
            SELECT COUNT(*) as count FROM chat_messages 
            WHERE chat_id = %s AND role = %s AND message = %s 
            AND timestamp > NOW() - INTERVAL 5 SECOND
        """
        result = execute_query(connection, check_query, (chat_id, role, message), fetch=True)
        
        if result[0]['count'] > 0:
            logging.debug(f"Skipping duplicate message for chat ID: {chat_id}")
            return True
            
        # Insert message
        query = """
            INSERT INTO chat_messages (chat_id, role, message, sql_query, visualization, timestamp) 
            VALUES (%s, %s, %s, %s, %s, NOW())
        """
        visualization_json = json.dumps(visualization) if visualization else None
        execute_query(connection, query, (chat_id, role, message, sql_query, visualization_json), fetch=False)
        return True
    except Exception as e:
        logging.error(f"Error saving chat message: {e}")
        return False
    finally:
        if connection and connection.is_connected():
            connection.close()

# Get chat history from database
def get_chat_history_from_db(chat_id):
    """Retrieve chat history for a specific chat ID."""
    connection = create_db_connection()
    if connection is None:
        logging.error("Failed to retrieve chat history: No database connection")
        return []
    
    try:
        query = """
            SELECT role, message, sql_query, visualization, timestamp 
            FROM chat_messages 
            WHERE chat_id = %s 
            ORDER BY timestamp ASC
        """
        results = execute_query(connection, query, (chat_id,), fetch=True)
        # Parse visualization JSON
        for result in results:
            if result['visualization']:
                result['visualization'] = json.loads(result['visualization'])
        return results
    except Exception as e:
        logging.error(f"Error retrieving chat history: {e}")
        return []
    finally:
        if connection and connection.is_connected():
            connection.close()

# Mark chat session as deleted
def mark_chat_as_deleted(chat_id):
    """Mark a chat session as deleted."""
    connection = create_db_connection()
    if connection is None:
        logging.error("Failed to mark chat as deleted: No database connection")
        return False
    
    try:
        query = "UPDATE chat_sessions SET deleted = TRUE WHERE id = %s"
        execute_query(connection, query, (chat_id,), fetch=False)
        return True
    except Exception as e:
        logging.error(f"Error marking chat as deleted: {e}")
        return False
    finally:
        if connection and connection.is_connected():
            connection.close()

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
def format_conversation_history(chat_id):
    """Format recent conversation history for context in prompts."""
    # Fetch history from database
    history = get_chat_history_from_db(chat_id)
    
    # Get the last CONTEXT_WINDOW_SIZE exchanges (5 pairs = 10 messages)
    recent_history = history[-2*CONTEXT_WINDOW_SIZE:] if len(history) > 2*CONTEXT_WINDOW_SIZE else history
    
    # Format messages for context
    formatted_messages = []
    for entry in recent_history:
        role = "User" if entry['role'] == 'user' else "Assistant"
        formatted_messages.append(f"{role}: {entry['message'].strip()}")
    
    return "\n".join(formatted_messages)

# Determine if query is a continuation
def is_query_continuation(user_query, chat_id):
    """Determine if the current query is a continuation of prior queries."""
    formatted_history = format_conversation_history(chat_id)
    if not formatted_history:
        logging.debug("No history available, treating query as standalone")
        return False
    
    # Check for explicit standalone indicators
    standalone_indicators = ['now show me', 'give me a new', 'all customers', 'every customer']
    if any(indicator in user_query.lower() for indicator in standalone_indicators):
        logging.debug("Standalone indicator detected, treating query as new")
        return False
    
    # Check for vague or follow-up phrases
    follow_up_indicators = ['how many are', 'what about', 'and', 'also', 'now', 'ok', 'same']
    is_vague = any(user_query.lower().startswith(phrase) for phrase in follow_up_indicators)
    
    # Use LLM to analyze semantic continuity
    prompt = f"""
You are a context-aware assistant that decides whether a user's current query is a continuation of the prior conversation or a new, standalone question.
 
A query is a **continuation** if:
- It refers to the **same dataset, topic, or filter** as the previous query (e.g., still discussing a specific state or policy type).
- It omits context that was clearly stated earlier (e.g., says “now group by gender” after a question about Arizona).
- It includes vague follow-ups like “same”, “also”, “what about now”, or relies on prior filters to be understood.
 
A query is **not a continuation** if:
- It **explicitly includes all the details** required to answer it independently (i.e., it's fully self-contained).
- It introduces a **new topic, filter, grouping, or entity**, such as switching from “state” to “marital status”, or from “Arizona” to “all customers”.
- It uses phrases like “start over”, “all data”, or shifts to a new scope.
 
You must evaluate the **semantic dependency** between queries, not just keyword matching.
 
Return only: `True` or `False`
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
            if 'from california' in question:
                filters['State'] = 'California'
            elif 'males' in question or 'male' in question:
                filters['Gender'] = 'M'
            elif 'females' in question or 'female' in question:
                filters['Gender'] = 'F'
    
    logging.debug(f"Extracted filters: {filters}")
    return filters

# Detect if query is requesting a customer list
def is_customer_list_query(user_query):
    """Determine if the query is asking for a list of customers."""
    list_indicators = [
        r"\bwhat\s+are\s+the\s+customers\b",
        r"\blist\s+customers\b",
        r"\bshow\s+customers\b",
        r"\bwho\s+are\s+the\s+customers\b",
        r"\bcustomers\s+who\b"
    ]
    return any(re.search(pattern, user_query.lower()) for pattern in list_indicators)

# SQL generation function
def generate_sql(user_query, schema_description, chat_id):
    """Generates SQL query using LLM with improved handling of percentage queries, group-by clauses, time-based filtering, and consistent handling of 'top N' queries."""
    # Get formatted conversation history
    formatted_history = format_conversation_history(chat_id)
    
    # Determine if the query is a continuation
    use_context = is_query_continuation(user_query, chat_id)
    
    logging.debug(f"Query continuation: {use_context}")
    logging.debug(f"User query: {user_query}")

    # Detect 'top N' queries and extract limit
    top_n_match = re.search(r'\b(top|highest|most)\s*(\d+)?\b', user_query.lower())
    is_top_n = bool(top_n_match)
    top_n_limit = int(top_n_match.group(2)) if top_n_match and top_n_match.group(2) else 3
    
    # Check if it's a customer list query
    is_list_query = is_customer_list_query(user_query)
    
    # Get prior SQL from chat history for filter extraction
    prior_sql = None
    history = get_chat_history_from_db(chat_id)
    for entry in history[::-1]:
        if entry.get('sql_query') and not entry['sql_query'].startswith('Answer from history:'):
            prior_sql = entry['sql_query']
            break
    
    # Extract filters if continuation
    filters = extract_filters_from_history(formatted_history, prior_sql) if use_context else {}
    
    # Check for time-related phrases
    has_time_reference = re.search(r'\b(year|years|month|months|day|days|recent|latest|last|past|since)\b', user_query.lower())
    
    # Check for dimension/category breakdown requests
    group_by_match = re.search(r'\b(?:by|per|across|for each|grouped by|break(?:ing)? down by)\s+([a-zA-Z\s]+)\b', user_query.lower())
    needs_group_by = bool(group_by_match)
    
    # Check for percentage queries
    is_percentage_query = re.search(r'\b(percentage|percent|proportion)\b', user_query.lower())
    
    # Enhanced time filtering guidance
    time_filtering_guide = """
TIME-BASED FILTERING RULES:
1. For any query with time references (e.g., "last X years", "past X months", "recent"):
   - PREFERRED: Use "Months Since Last Claim" for claim recency (e.g., WHERE "Months Since Last Claim" <= 24 for last 2 years)
   - PREFERRED: Use "Months Since Policy Inception" for policy age/duration (e.g., WHERE "Months Since Policy Inception" <= 24 for policies started in last 2 years)
   - ALTERNATIVE: If "Effective To Date" exists, use it for active policy status (WHERE "Effective To Date" >= current_date)
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
1. For queries mentioning "by [dimension]" (e.g., "by gender", "by state"):
   - Include the dimension in the SELECT clause
   - Add a GROUP BY clause for that dimension
   - Example: "average claim amount by gender" → 
     SELECT "Gender", AVG("Total Claim Amount") AS avg_claim_amount FROM insurance GROUP BY "Gender";
2. For multiple dimensions:
   - Include all dimensions in both SELECT and GROUP BY
   - Example: "claims by gender and state" → 
     SELECT "Gender", "State", COUNT(*) as claim_count FROM insurance GROUP BY "Gender", "State";
3. For queries with filtering and group by:
   - Apply filters with WHERE clause, then GROUP BY
   - Example: "average claim amount by gender from Arizona" → 
     SELECT "Gender", AVG("Total Claim Amount") AS avg_claim_amount FROM insurance WHERE "State" = 'Arizona' GROUP BY "Gender";
4. For "by [dimension]" requests:
   - Do NOT filter the dimension unless requested
"""

    percentage_guide = """
PERCENTAGE QUERY RULES:
1. For queries asking for percentage/proportion of a category (e.g., "percentage of claims for Two-Door Car"):
   - Calculate: (COUNT of rows matching the condition) * 100.0 / (TOTAL COUNT of all rows)
   - Use a subquery for the total count: (SELECT COUNT(*) FROM insurance)
   - Example: "how many percentage of claim for Two-Door Car"
     Correct SQL: 
     SELECT (COUNT(*) * 100.0 / (SELECT COUNT(*) FROM insurance)) AS "Percentage of Claims"
     FROM insurance
     WHERE "Vehicle Class" = 'Two-Door Car';
2. Ensure the denominator is the TOTAL number of rows, not the filtered count.
3. Apply any additional filters from the query or context.
4. Use a clear alias like "Percentage of Claims" for the result.
5. For queries asking for percentage/proportion of a category (e.g., "How much percentage of claims by males with different education level from Nevada"):
   - Calculate: (COUNT of rows matching each subgroup condition) * 100.0 / (TOTAL COUNT of rows matching the main condition)
   - Use a subquery for the total count of the filtered group:(SELECT COUNT(*) FROM insurance WHERE <main condition>)
   - Group by the desired category (e.g., "Education") to get the breakdown.
   - Example: "How much percentage of claims by males with different education level from Nevada"
     Correct SQL: 
     SELECT "Education",
       (COUNT(*) * 100.0 / (SELECT COUNT(*) FROM insurance WHERE "Gender" = 'M' AND "State" = 'Arizona')) AS "Percentage of Claims"
     FROM insurance
     WHERE "Gender" = 'M' AND "State" = 'Arizona'
     GROUP BY "Education";
"""

    prompt = f"""
DATABASE SCHEMA:
{schema_description}

NOTE:'Gender' has values 'M', 'F'. 'Coverage' has 'Basic', 'Extended', 'Premium'. 'Education' has 'Bachelor', 'High School or Below', 'College', 'Master', 'Doctor'.'Response' has 'No', 'Yes'. 'Vehicle Class' has ' Four-Door Cars', 'SUVs', 'Two-Door Cars', 'Sports Cars', 'Luxury Cars', 'Luxury SUVs'.'Employment Status' has 'Employed', 'Unemployed', 'Retired', 'Medical Leave', 'Disabled'.'Location' has 'Urban', 'Rural, 'Suburban'.'Marital Status' has 'Married', 'Single', 'Divorced'.'Policy Type' has 'Corporate Auto', 'Personal Auto', 'Special Auto'.'Policy' has 'Corporate L1', 'Corporate L2', 'Corporate L3', 'Personal L1', 'Personal L2', 'Personal L3', 'Special L1', 'Special L2', 'Special L3'.'Sales Channel' has 'Web', 'Branch', 'Agent', 'Call Center'.'Vehicle Size' has 'Medsize', 'Small', 'Large'.

CONVERSATION HISTORY:
{formatted_history if use_context else 'No relevant history for this query'}

PRIOR FILTERS (apply these if continuation):
{json.dumps(filters) if filters else 'None'}

{time_filtering_guide if has_time_reference else ''}

{group_by_guide if needs_group_by else ''}

{percentage_guide if is_percentage_query else ''}

CRITICAL INSTRUCTIONS:
1. CONTEXT AWARENESS:
   - For CONTINUATION queries, include ALL prior filters using AND clauses.
   - For vague queries, assume they apply to the prior context.
   - For standalone queries (e.g., 'now show me'), ignore prior context.
2. GROUP BY HANDLING:
   - For "by [dimension]" queries, use GROUP BY and include the dimension in SELECT.
   - Do NOT filter the grouped dimension unless requested.
3. TIME-BASED QUERIES:
   - Use "Months Since Last Claim" for claim recency, "Months Since Policy Inception" for policy age.
   - Convert years to months (e.g., 2 years = 24 months).
   - If "Effective To Date" is used, handle date functions like strftime('%Y', "Effective To Date") for year extraction.
4. TOP N / HIGHEST QUERIES:
   - Use ORDER BY DESC LIMIT {top_n_limit} for 'top', 'highest', 'most' queries.
   - Aggregate if multiple rows per entity (e.g., MAX("Total Claim Amount")).
5. CUSTOMER LIST QUERIES:
   - For queries like 'what are the customers married and located in California':
     - Select columns like "Customer", "State", "Marital Status"
     - Add LIMIT 10
6. PERCENTAGE QUERIES:
   - For queries like 'percentage of claims for Two-Door Car':
     - Use: (COUNT with condition) * 100.0 / (SELECT COUNT(*) FROM insurance)
     - Example: 
       SELECT (COUNT(*) * 100.0 / (SELECT COUNT(*) FROM insurance)) AS "Percentage of Claims"
       FROM insurance
       WHERE "Vehicle Class" = 'Two-Door Car';
7. SQL GENERATION:
   -Use `SUM()` when the column represents a quantity per customer that should be totaled across rows:
     - Examples: "Number of Policies", "Number of Open Complaints", "Total Claim Amount"
   -Use `AVG()` when the goal is to find average values for continuous numeric fields:
     - Examples: "Income", "Customer Lifetime Value", "Monthly Premium Auto"
   - Use `COUNT(*)` only when the user explicitly asks for the **number of customers** or **number of records**, not for summing a numeric column.
   - Wrap column names with spaces in double quotes (e.g., "Vehicle Class").
   - Use precise values (e.g., "Vehicle Class" = 'Two-Door Car').
   - Return plain text SQL without markdown.
   - For customer list queries, limit to 10 rows and mention additional results.
8. EXAMPLES:
   - Query: 'percentage of claims for Two-Door Car'
     SELECT (COUNT(*) * 100.0 / (SELECT COUNT(*) FROM insurance)) AS "Percentage of Claims"
     FROM insurance
     WHERE "Vehicle Class" = 'Two-Door Car';
   - Query: 'what are the customers married and located in California?'
     SELECT "Customer", "State", "Marital Status"
     FROM insurance
     WHERE "State" = 'California' AND "Marital Status" = 'Married'
     LIMIT 10;
   - Query: 'claims by year in Arizona'
     SELECT "State", strftime('%Y', "Effective To Date") AS "Year", COUNT(*) AS "Total Claims"
     FROM insurance
     WHERE "State" = 'Arizona'
     GROUP BY "State", strftime('%Y', "Effective To Date");

CURRENT USER QUESTION: {user_query}

Generate ONLY the appropriate SQL query for SQLite as plain text without markdown, unless the answer is from history, then return 'Answer from history: [answer]'.
"""

    response = llm.invoke(prompt)
    sql = response.content.strip()
    
    # Post-process to remove markdown
    sql = re.sub(r'```sql\n|```', '', sql).strip()
    
    # Add LIMIT 10 for customer list queries if not present
    if is_list_query and 'LIMIT' not in sql.upper():
        sql = f"{sql} LIMIT 10"
    
    logging.debug(f"Generated SQL: {sql}")
    return sql

# SQL execution function
def run_sql(sql_query, engine):
    """Execute SQL and format results, including total count for customer list queries."""
    with engine.connect() as conn:
        # Check if it's a customer list query
        is_list_query = 'LIMIT 10' in sql_query.upper() and 'SELECT "Customer"' in sql_query
        
        if is_list_query:
            # Create a count query to get total matching rows
            count_query = sql_query.replace('SELECT "Customer"', 'SELECT COUNT(*) AS total_count')
            count_query = count_query.replace('LIMIT 10', '')
            count_result = conn.execute(text(count_query)).fetchone()
            total_count = count_result[0] if count_result else 0
        else:
            total_count = None
        
        # Execute the original query
        result = conn.execute(text(sql_query))
        rows = result.fetchall()
        columns = result.keys()

        if len(rows) == 1 and len(columns) == 1:
            return {"data": str(rows[0][0]), "total_count": total_count}
        elif len(rows) == 1:
            return {"data": dict(zip(columns, rows[0])), "total_count": total_count}
        else:
            data = [dict(zip(columns, row)) for row in rows]
            return {"data": data, "total_count": total_count}

# Analyze data structure
def analyze_data_structure(result_data):
    if isinstance(result_data, dict) and "data" in result_data:
        data_to_analyze = result_data["data"]
    else:
        data_to_analyze = result_data
 
    if not data_to_analyze:
        return False, "No data available for visualization"
 
    if isinstance(data_to_analyze, str) and data_to_analyze.replace('.', '').isdigit():
        return False, "Single numeric value is not suitable for visualization"
 
    if isinstance(data_to_analyze, list):
        if not data_to_analyze or len(data_to_analyze) < 2:
            return False, "Not enough data points for visualization"
 
        sample = data_to_analyze[0]
        if not isinstance(sample, dict):
            return False, "Unsupported data format"
 
        numeric_keys = []
        non_numeric_keys = []
        for key, value in sample.items():
            if value is None:
                continue
            try:
                if isinstance(value, (int, float)) or (isinstance(value, str) and value.replace(',', '').replace('.', '').isdigit()):
                    numeric_keys.append(key)
                else:
                    non_numeric_keys.append(key)
            except Exception:
                continue
 
        if not numeric_keys:
            return False, "No numeric values found for visualization"
 
        if len(non_numeric_keys) > 1:
            return False, f"Too many grouping fields ({', '.join(non_numeric_keys)}); unclear which one to use as x-axis or legend"
 
        return True, f"Data is suitable for x: {non_numeric_keys[0]}, y: {numeric_keys[0]}"
 
    return False, "Data structure not suitable for visualization"

# Identify axes
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
        
    keys = list(items[0].keys())
    
    numeric_keys = []
    categorical_keys = []
    
    for key in keys:
        all_numeric = True
        values = set()
        
        for item in items:
            value = item.get(key)
            values.add(str(value) if value is not None else '')
            
            try:
                if value is None:
                    all_numeric = False
                    continue
                if isinstance(value, (int, float)):
                    continue
                if isinstance(value, str) and value.replace(',', '').replace('.', '').isdigit():
                    continue
                all_numeric = False
            except AttributeError as e:
                logging.debug(f"Error checking value for key {key}: {value}, error: {e}")
                all_numeric = False
                
        if len(values) < 15 and not all_numeric:
            categorical_keys.append(key)
        elif all_numeric:
            numeric_keys.append(key)
            
    if categorical_keys and numeric_keys:
        preferred_x = ["Income", "State", "Gender", "Marital Status", "Vehicle Class", "Coverage", "Year"]
        for preferred in preferred_x:
            if preferred in categorical_keys:
                return preferred, numeric_keys[0]
        return categorical_keys[0], numeric_keys[0]
    
    elif categorical_keys:
        count_columns = [k for k in keys if 'count' in k.lower() or 'total' in k.lower() or 'avg' in k.lower()]
        if count_columns:
            return categorical_keys[0], count_columns[0]
        else:
            return categorical_keys[0], "COUNT(*)"
    
    elif len(numeric_keys) >= 2:
        if "Income" in numeric_keys:
            return "Income", [k for k in numeric_keys if k != "Income"][0]
        return numeric_keys[0], numeric_keys[1]
    
    elif len(numeric_keys) == 1:
        return "Index", numeric_keys[0]
    
    return None, None

# Determine if visualization is appropriate
def should_visualize(user_query, result_data):
    """Determine if visualization is appropriate based on query and data."""
    viz_terms = ["chart", "graph", "plot", "visualize", "visualization", "display"]
    explicit_request = any(term in user_query.lower() for term in viz_terms)
    
    suitable, reason = analyze_data_structure(result_data)
    
    if explicit_request and not suitable:
        return False, f"I can't create a visualization because: {reason}"
    elif explicit_request and suitable:
        return True, "User requested visualization and data is suitable"
    elif suitable:
        x_axis, y_axis = identify_axes(result_data)
        if x_axis and y_axis:
            return True, "Data structure is suitable for visualization"
        else:
            return False, "Data structure doesn't have clear axes for visualization"
    else:
        return False, "Data not suitable for visualization"

# Prepare visualization data
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
        chart_data = chart_data[:15]
    
    x_axis, y_axis = identify_axes(chart_data)
    
    if not x_axis or not y_axis:
        return None, None, None
    
    if y_axis == "COUNT(*)":
        count_field = "Count"
        for item in chart_data:
            item[count_field] = 1
        y_axis = count_field
    
    formatted_data = []
    for item in chart_data:
        x_value = item.get(x_axis)
        y_value = item.get(y_axis)
        if x_value is not None and y_value is not None:
            formatted_data.append({
                "x": str(x_value),
                "y": float(y_value) if isinstance(y_value, (int, float)) else float(y_value.replace(',', '')) if isinstance(y_value, str) and y_value.replace(',', '').replace('.', '').isdigit() else 0
            })
    
    return formatted_data, x_axis, y_axis

# Determine chart type
def determine_chart_type(user_query, chart_data, x_axis, y_axis):
    """Determine the best chart type based on data characteristics."""
    query = user_query.lower()
    
    if re.search(r'\b(bar|column)\s*(chart|graph|plot|viz)', query):
        return "bar"
    elif re.search(r'\b(pie|circle|donut)\s*(chart|graph|plot|viz)', query):
        return "pie"
    elif re.search(r'\b(line|trend|time)\s*(chart|graph|plot|viz)', query):
        return "line"
    elif re.search(r'\b(scatter|point|dots)\s*(chart|graph|plot|viz)', query):
        return "scatter"
    
    if not chart_data or not x_axis or not y_axis:
        return "bar"
    
    unique_x_values = set(item["x"] for item in chart_data)
    
    time_related = any(term in x_axis.lower() for term in ["date", "time", "year", "month", "day"])
    
    is_percentage = "percentage" in y_axis.lower() or all(float(item["y"]) <= 100 for item in chart_data)
    
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

# Here I am I have to modified above code and add such funciton which will display the remaining data as well
import json
import re

def generate_user_friendly_response(user_query, result_data, sql_query, has_visualization=False):
    """Generates a conversational, user-friendly response to data queries."""
    is_list_query = is_customer_list_query(user_query)
    is_percentage_query = re.search(r'\b(percentage|percent|proportion)\b', user_query.lower())

    print(f"Result data: {result_data}")

    if isinstance(result_data, dict) and "data" in result_data:
        data = result_data["data"]
        total_count = result_data.get("total_count")
        print(f"Total count: {total_count}")
    else:
        print("No data found")
        data = result_data
        total_count = None

    if isinstance(data, list):
        print("Data is a list")
        # Allow full data for grouped summary queries (e.g., grouped by State and Education)
        if is_list_query:
            result_summary = json.dumps(data[:10], indent=2)
        else:
            result_summary = json.dumps(data, indent=2)
        more_count = total_count - len(data) if total_count and total_count > len(data) else 0
    elif isinstance(data, dict):
        print("Data is a dictionary")
        result_summary = json.dumps(data, indent=2)
        more_count = 0
    else:
        print("Data is not a list or dict")
        result_summary = str(data)
        more_count = 0

    prompt = f"""
Create a conversational, friendly response to the user's question using the data results.

User question: {user_query}
SQL query executed: {sql_query}
Data result: {result_summary}
Additional results not shown: {more_count}
Has visualization: {has_visualization}

Guidelines:
1. Give clear and friendly responses, like a helpful assistant.
2. Address the user's question directly and completely.
3. For customer list queries (e.g., 'what are the customers married and located in California?'):
   - List customer details (e.g., Customer ID, State, Marital Status) concisely.
   - Do NOT summarize unrelated metrics unless requested.
   - If more_count > 0, mention 'X more results available'.
   - Example: "Here are the top 10 married customers from California: [list]. There are 3010 more results available."
4. For percentage queries (e.g., 'percentage of claims for Two-Door Car'):
   - Clearly state the percentage with precision (e.g., "23.5% of claims are for Two-Door Cars").
   - Avoid misleading statements (e.g., don't say "all claims" unless 100%).
   - If the percentage is 0%, suggest there may be no matching records.
   - Example: "About 23.5% of claims are for Two-Door Cars."
5. For other queries, highlight key insights from the data.
6. Use natural language, avoiding technical terms.
7. Keep answers concise but complete (2-3 sentences for non-list queries).
8. If there's a visualization, mention it briefly.
9. Be precise with numerical results.

Example for percentage query:
User question: "how many percentage of claim for Two-Door Car"
Response: "About 23.5% of claims are for Two-Door Cars. Let me know if you want more details!"

Example for grouped query:
User question: "claims by year in Arizona"
Response: "In Arizona, claim counts by year are: 2020 (351 claims), 2021 (352 claims), 2022 (325 claims), 2023 (355 claims), and 2024 (320 claims). I've also created a chart to show this trend!"

Be friendly and focus on answering the user's question accurately.
"""

    response = llm.invoke(prompt)
    print(f"LLM Response:\n{response.content}")
    return response.content.strip()


# Main database query function
def ask_question(user_query, schema, chat_id):
    """Processes database queries with direct visualization handling."""
    global current_query_context
    try:
        engine = setup_database()

        with get_openai_callback() as cb:
            print("HERE I AM")
            print(f"Chat ID: {chat_id}, User Query: {user_query}")
            sql_query = generate_sql(user_query, schema, chat_id)
            
            if sql_query.startswith("Answer from history:"):
                print("Using cached response from history")
                response_text = sql_query.replace("Answer from history:", "").strip()
                result_data = None
                visualization_data = None
                current_query_context = QueryContext()
            else:
                result_data = run_sql(sql_query, engine)
                
                current_query_context.sql = sql_query
                current_query_context.result = result_data
                current_query_context.columns = list(result_data["data"][0].keys()) if isinstance(result_data["data"], list) and result_data["data"] else None
                current_query_context.filters = extract_filters_from_history(format_conversation_history(chat_id), sql_query)
                
                should_viz, viz_reason = should_visualize(user_query, result_data)
                
            
                print(f"Should visualize: {should_viz},")
                
                visualization_data = None
                if should_viz:
                    chart_data, x_axis, y_axis = prepare_visualization_data(result_data)
                    
                    if chart_data and x_axis and y_axis:
                        chart_type = determine_chart_type(user_query, chart_data, x_axis, y_axis)
                        
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
                        current_query_context.x_axis = x_axis
                        current_query_context.y_axis = y_axis
                
                response_text = generate_user_friendly_response(user_query, result_data, sql_query, should_viz)
                
                if any(term in user_query.lower() for term in ["chart", "graph", "visual"]) and not should_viz:
                    response_text += f" {viz_reason}"
        
            save_chat_message(chat_id, 'assistant', response_text, sql_query, visualization_data)
            
            return {
                "text": response_text,
                "visualization": visualization_data,
                "sql": sql_query
            }
    except Exception as e:
        logging.error(f"Error in ask_question: {str(e)}")
        response_text = "I can assist only with questions that are derived from the customer dataset — such as policies, claims, coverage, and customer insights"
        
        save_chat_message(chat_id, 'assistant', response_text)
        return {"text": response_text, "visualization": None, "sql": None}

# General query function
def ask_general_question(query, chat_id, bot_name="Aira"):
    """Handles general non-database queries with natural responses."""
    try:
        formatted_history = format_conversation_history(chat_id)
        query_lower = query.lower()
        
        if any(phrase in query_lower for phrase in ["what's your name", "what is your name", 
                                                    "who are you", "your name", "what should i call you"]):
            name_response = f"My name is {bot_name}! I'm your insurance assistant, ready to help you analyze insurance data and answer any questions you might have."
            
            save_chat_message(chat_id, 'assistant', name_response)
            return {"text": name_response, "visualization": None, "sql": None}
        
        if any(phrase in query_lower for phrase in ["what can you do", "your capabilities", "what are you able to",
                                                   "help me with", "what do you know", "how can you help",
                                                   "what can you help with", "what are your abilities"]):
            capability_response = f"""I'm {bot_name}, here to help you gain valuable insights from your insurance customers database. You can ask about claim amounts, customer demographics, policy types, vehicle details, coverage patterns, and sales channel trends. I deliver clear, data-driven responses based on records."""
            
            save_chat_message(chat_id, 'assistant', capability_response)
            return {"text": capability_response, "visualization": None, "sql": None}
        
        is_greeting = any(greeting in query_lower for greeting in ["hello", "hi ", "hey", "good morning", "good afternoon",
                                                                  "good evening", "howdy", "what's up", "greetings"])
        
        if is_greeting:
            greeting_prompt = f"""
You are {bot_name}, insurance assistant responding to a greeting.

USER GREETING: {query}

Guidelines for your response:
- Introduce yourself as {bot_name}
- Respond with a personalized greeting that matches the user's tone and time-of-day reference (if any)
- If the user says "Good morning", respond with an enhanced morning greeting
- If the user says "Good afternoon", respond with an enhanced afternoon greeting
- If the user says "Good evening", respond with an enhanced evening greeting
- Keep it brief and warm (1-2 sentences)
- Mention you're an Insurance data insights assistant ready to help
- Include that you can help in exploring customer-level data on policies, claims, and customer value to uncover meaningful trends and patterns, all based strictly on the available dataset.
- Be conversational and natural
- Avoid sounding robotic

Example responses:
- For "Good morning": "A wonderful morning to you! I'm {bot_name}, your insurance assistant ready to provide insights and answer your insurance questions."
- For "Hi there": "Hi there! I'm {bot_name}, your insurance assistant, ready to dive into insurance data and answer any questions you might have."

Generate ONLY the greeting response, nothing else.
"""
            greeting_response = llm.invoke(greeting_prompt).content.strip()
            
            save_chat_message(chat_id, 'assistant', greeting_response)
            return {"text": greeting_response, "visualization": None, "sql": None}
        
        prompt = f"""
You are {bot_name},  insurance assistant having a conversation. You specialize in explaining insurance concepts and providing helpful information.

CONVERSATION HISTORY:
{formatted_history}

CURRENT QUESTION:
{query}

Guidelines for your response:
- Always refer to yourself as {bot_name} if you need to mention your name
- Be warm and conversational&nbsp;And make sure to keep the conversation history in mind
- If the question is about an insurance concept or definition (like "What is a deductible?", "How does comprehensive coverage work?"), provide a CLEAR and CONCISE explanation
- For general insurance questions, focus on being INFORMATIVE while using simple language
- For greetings or social messages, be BRIEF and FRIENDLY
- Keep your answers focused on auto insurance topics when relevant
- Avoid insurance jargon when possible, or explain it immediately if you use it
- Be concise but helpful (2-4 sentences is ideal)
- Sound human and conversational, not like a technical manual
"""
        response = llm.invoke(prompt)
        response_text = response.content.strip()
        
        save_chat_message(chat_id, 'assistant', response_text)
        return {"text": response_text, "visualization": None, "sql": None}
    except Exception as e:
        logging.error(f"Error in ask_general_question: {str(e)}")
        response_text = "I'm sorry, I couldn't process that. Could you try again?"
        
        save_chat_message(chat_id, 'assistant', response_text)
        return {"text": response_text, "visualization": None, "sql": None}


def classify_query(user_query, schema, chat_id):
    """Uses LLM to classify the intent of the query with context-aware logic."""
    try:
        formatted_history = format_conversation_history(chat_id)
        save_chat_message(chat_id, 'user', user_query)
        classification_prompt = f"""
You are an intent classifier for an insurance assistant.

CONVERSATION HISTORY:
{formatted_history}

USER QUERY TO CLASSIFY:
"{user_query}"

Your job is to classify the query into ONE of these categories:

1. **data_query** – The user is asking about:
   - Customer data, insurance policies, claims, premiums, payments from the database
   - Statistics/aggregations about the data (counts, averages, percentages, etc.)
   - Simple listing requests like "show me claims by state" without visualization format specified
   - Follow-up questions that logically extend previous data queries
   - Examples: "How many customers are married?", "What about California customers?", "Percentage of claims for Two-Door Car", "Show me state-wise claim data"
   - CRITICAL: "What is insurance?" is NOT a data query
   - CRITICAL: "What is auto insurance?" is NOT a data query
   - CRITICAL: "What can you do?" is NOT a data query
   - CRITICAL: "What are your capabilities?" is NOT a data query
   - CRITICAL: Any question about the chatbot itself rather than insurance data is NOT a data query

2. **general_chat** – The user is asking about:
   - Capabilities of the chatbot (IMPORTANT: ALL capability questions go here)
   - Greetings, thanks, closings, help requests
   - MUST INCLUDE: "What can you do?", "What are your capabilities?", "What do you know?", "How can you help me?"
   - Other examples: "Hello", "Thanks", "Goodbye", "Help me", "Tell me what you can assist with"
   - ANY question about what the chatbot can or cannot do belongs in this category

3. **out_of_scope** – Completely unrelated to auto insurance:
   - CRITICAL: Asking "what is auto insurance?" or "what is insurance?" is out of scope
   - Weather, sports, news, other insurance types
   - CRITICAL: Examples: "What's the weather?", "Who won the World Cup?", "Tell me about health insurance", "What is auto insurance?", "What is insurance?"

4. **visualization_only** – Request to visualize previously retrieved or discussed data:
   - User first requested data and is now asking for visualization without new criteria
   - Examples: "Make a pie chart of this data", "Visualize the data", "Show this as a bar chart", "Create a graph for these results"
   - CRITICAL: These are follow-up requests to visualize data already discussed

5. **visualization_with_criteria** – Request for visualization with specific data criteria:
   - Direct requests for visualizations with filtering criteria in a single query
   - Examples: "Show me a bar chart of claims by state", "Create a pie chart for total claim amount by vehicle type", "Give me visualization for total claim amount by state", "Graph the average premium by age group"
   - CRITICAL: Any request that simultaneously asks for visualization AND specifies what data to visualize

CLASSIFICATION RULES:
1. CAPABILITY QUESTIONS: Any questions about what the chatbot can do, its abilities, or how it can help MUST be classified as "general_chat"
2. VISUALIZATION DETECTION: If the query includes terms like "chart", "graph", "plot", "visualization", "pie chart", "bar graph" AND specifies what data to visualize, classify as "visualization_with_criteria"
3. FOLLOW-UP DETECTION: If user previously requested data and now only asks for visualization format (without specifying new data criteria), classify as "visualization_only"
4. DATA QUESTIONS: Queries about insurance database information without visualization requests are "data_query"
5. INFORMATIONAL QUESTIONS: General questions about insurance concepts (not database queries) are "out_of_scope"

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
            return "data_query"
    except Exception as e:
        logging.error(f"Classification error: {str(e)}")
        return "general_chat"

# Handle visualization only
def handle_visualization_only(user_query, chat_id):
    """Handles requests to visualize the most recent data query result."""
    global current_query_context
    
    if not current_query_context.result:
        response_text = "I don't have any recent data to visualize. Please provide a data query first."
        save_chat_message(chat_id, 'assistant', response_text)
        return {
            "text": response_text,
            "visualization": None,
            "sql": None
        }

    result_data = current_query_context.result
    chart_data, x_axis, y_axis = prepare_visualization_data(result_data)
    
    if not chart_data or not x_axis or not y_axis:
        response_text = "The recent data doesn't have a structure I can visualize effectively. Could you try a different query that returns data with categories and numbers?"
        save_chat_message(chat_id, 'assistant', response_text)
        return {
            "text": response_text,
            "visualization": None,
            "sql": None
        }

    chart_type = determine_chart_type(user_query, chart_data, x_axis, y_axis)
    
    visualization_data = {
        "type": chart_type,
        "data": chart_data,
        "x_axis": x_axis,
        "y_axis": y_axis,
        "x_label": x_axis.replace("_", " ").title(),
        "y_label": y_axis.replace("_", " ").title()
    }

    if chart_type == "bar":
        response_text = f"Here's a bar chart showing {y_axis.replace('_', ' ').title()} across different {x_axis.replace('_', ' ').title()} values."
    elif chart_type == "pie":
        response_text = f"Here's a pie chart showing the distribution of {y_axis.replace('_', ' ').title()} across {x_axis.replace('_', ' ').title()} categories."
    elif chart_type == "line":
        response_text = f"Here's a line chart showing how {y_axis.replace('_', ' ').title()} changes across {x_axis.replace('_', ' ').title()} values."
    else:
        response_text = f"Here's a {chart_type} chart visualization of the recent data."
        
    save_chat_message(chat_id, 'assistant', response_text, current_query_context.sql, visualization_data)
    return {
        "text": response_text,
        "visualization": visualization_data,
        "sql": None
    }

# Out of scope handler
def handle_out_of_scope(query, chat_id):
    """Handles questions that are not relevant to insurance."""
    response_text = "I can assist only with questions that are derived from the customer dataset — such as policies, claims, coverage, and customer insights"
    
    save_chat_message(chat_id, 'assistant', response_text)
    return {"text": response_text, "visualization": None, "sql": None}

# Main query handler
def handle_query(user_query, schema, chat_id):
    """Routes the query to the correct agent based on classification."""
    query_type = classify_query(user_query, schema, chat_id)
    logging.debug(f"Query type detected: {query_type} amd user query: {user_query}")

    if query_type == "visualization_only":
        return handle_visualization_only(user_query, chat_id)
    elif query_type == "visualization_with_criteria":
        return ask_question(user_query, schema, chat_id)
    elif query_type == "data_query":
        return ask_question(user_query, schema, chat_id)
    elif query_type == "general_chat":
        return ask_general_question(user_query, chat_id)
    else:
        return handle_out_of_scope(user_query, chat_id)

# Process user input with session management
def process_user_input_logic(user_query, client_id=None, chat_id=None):
    """Process user input with session management."""
    if chat_id:
        pass
    elif client_id:
        chat_id = get_or_create_chat_session_by_client_id(client_id)
    else:
        return {"status": "error", "message": "Either chat_id or client_id must be provided"}
    
    engine = setup_database()
    schema = get_schema_description(engine)
    
    response = handle_query(user_query, schema, chat_id)
    response['chat_id'] = chat_id
    return response

# API Endpoints
@app.route("/api/chat", methods=["POST"])
def chat():
    """Handle chat messages with session management."""
    data = request.json
    user_message = data.get("message", "")
    chat_id = data.get("chat_id")
    client_id = data.get("client_id", get_client_ip())

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    response = process_user_input_logic(user_message, client_id, chat_id)
    return jsonify({
        "response": response["text"],
        "visualization": response["visualization"],
        "sql": response["sql"],
        "chat_id": response["chat_id"]
    })

@app.route('/api/chat/session', methods=['POST'])
def get_chat_session():
    """Get or create a chat session for the provided client ID."""
    try:
        data = request.json
        client_id = data.get('client_id', get_client_ip())
        
        if not client_id:
            return jsonify({
                "status": "error",
                "message": "Client ID is required"
            }), 400
            
        chat_id = get_or_create_chat_session_by_client_id(client_id)
        
        return jsonify({
            "status": "success",
            "chat_id": chat_id
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/chat/history/<chat_id>', methods=['GET'])
def get_chat_history_api(chat_id):
    """Retrieve chat history for a specific chat ID."""
    try:
        history = get_chat_history_from_db(chat_id)
        formatted_history = []
        
        for msg in history:
            formatted_history.append({
                "role": msg['role'],
                "content": msg['message'],
                "sql": msg['sql_query'],
                "visualization": msg['visualization'],
                "timestamp": msg['timestamp'].isoformat() if isinstance(msg['timestamp'], datetime.datetime) else msg['timestamp']
            })
        
        return jsonify({
            "status": "success",
            "chat_id": chat_id,
            "history": formatted_history
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/chat/<chat_id>/delete', methods=['POST'])
def delete_chat_session(chat_id):
    """Mark a chat session as deleted."""
    try:
        if mark_chat_as_deleted(chat_id):
            return jsonify({
                "status": "success",
                "message": "Chat session deleted successfully"
            })
        else:
            return jsonify({
                "status": "error",
                "message": "Failed to delete chat session"
            }), 500
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"})

@app.route("/api/reset", methods=["POST"])
def reset_conversation():
    """Reset conversation history for a specific chat_id."""
    global current_query_context
    data = request.json
    chat_id = data.get("chat_id")
    
    if not chat_id:
        return jsonify({"status": "error", "message": "chat_id is required"}), 400
    
    if mark_chat_as_deleted(chat_id):
        current_query_context = QueryContext()
        logging.info(f"Conversation history and query context reset for chat_id: {chat_id}")
        return jsonify({"status": "success", "message": "Conversation reset"})
    else:
        return jsonify({"status": "error", "message": "Failed to reset conversation"}), 500

if __name__ == "__main__":
    app.run(debug=True, host="192.168.1.14", port=5000)