from flask import Flask, request, jsonify,json
import re
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langsmith import traceable
import mysql.connector
from mysql.connector import Error
import uuid
import random
from datetime import datetime
from dateutil import parser
from flask_cors import CORS
import time

app = Flask(__name__)
CORS(app)

current_time = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
current_year = datetime.now().year

load_dotenv()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAISS_PATH = os.path.join(BASE_DIR, "faiss_db")
GREETINGS = ["hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening"]

# Session store for multiple users
session_store = {}

# Session state class
class SessionState:
    def __init__(self):
        self.chat_history = []
        self.db_configured = False
        self.db_params = {
            "host": os.getenv("DB_HOST", "127.0.0.1"),
            "port": os.getenv("DB_PORT", "3306"),
            "database": os.getenv("DB_NAME", ""),
            "user": os.getenv("DB_USER", ""),
            "password": os.getenv("DB_PASSWORD", ""),
            "use_ssl": False
        }
        self.product_context = None
        self.last_order_id = None
        self.last_sql_results = None
        self.last_sql_query = None
        self.last_intent = None
        self.query = None

def get_session_state(session_id):
    """Retrieve or create session state for a given session ID."""
    if session_id not in session_store:
        session_store[session_id] = SessionState()
    return session_store[session_id]

# Initialize embeddings
def get_embeddings():
    return OpenAIEmbeddings()

def test_db_connection(session_id, max_retries=3, retry_delay=2):
    """Test database connection with retries."""
    session_state = get_session_state(session_id)
    for attempt in range(max_retries):
        try:
            conn = mysql.connector.connect(
                host=session_state.db_params["host"],
                port=session_state.db_params["port"],
                database=session_state.db_params["database"],
                user=session_state.db_params["user"],
                password=session_state.db_params["password"],
                ssl_disabled=not session_state.db_params["use_ssl"]
            )
            conn.close()
            return True
        except Error as e:
            print(f"Connection attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
    return False

def get_db_connection(session_id):
    """Establish a database connection."""
    session_state = get_session_state(session_id)
    try:
        conn = mysql.connector.connect(
            host=session_state.db_params["host"],
            port=session_state.db_params["port"],
            database=session_state.db_params["database"],
            user=session_state.db_params["user"],
            password=session_state.db_params["password"],
            ssl_disabled=not session_state.db_params["use_ssl"],
            connection_timeout=10
        )
        return conn
    except Error as e:
        print(f"Database connection error: {e}")
        return None

def validate_db_schema(session_id):
    """Validate that required tables and columns exist."""
    conn = get_db_connection(session_id)
    if not conn:
        return False, "Failed to connect to database."
    
    cursor = conn.cursor()
    try:
        # Check if chat_messages table exists with required columns
        cursor.execute("""
            SELECT COLUMN_NAME
            FROM information_schema.columns
            WHERE table_schema = DATABASE() AND table_name = 'chat_messages';
        """)
        columns = [row[0] for row in cursor.fetchall()]
        required_columns = ['id', 'chat_id', 'role', 'message', 'timestamp']
        missing_columns = [col for col in required_columns if col not in columns]
        
        if missing_columns:
            return False, f"Missing columns in chat_messages: {', '.join(missing_columns)}"
        
        return True, "Database schema validated successfully."
    except Error as e:
        return False, f"Error validating schema: {e}"
    finally:
        cursor.close()
        conn.close()

def get_db_schema(session_id):
    """Fetch database schema."""
    conn = get_db_connection(session_id)
    if not conn:
        return {"tables": {}, "error": "Failed to connect to database."}
    
    cursor = conn.cursor()
    schema = {"tables": {}}
    try:
        cursor.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = DATABASE();
        """)
        tables = [row[0] for row in cursor.fetchall()]
        
        for table in tables:
            schema["tables"][table] = {"columns": [], "foreign_keys": []}
            
            cursor.execute("""
                SELECT column_name, data_type, column_key
                FROM information_schema.columns
                WHERE table_name = %s;
            """, (table,))
            columns = cursor.fetchall()
            schema["tables"][table]["columns"] = [
                {"name": col[0], "data_type": col[1], "is_primary": col[2] == "PRI"}
                for col in columns
            ]
            
            try:
                cursor.execute("""
                    SELECT 
                        COLUMN_NAME, 
                        REFERENCED_TABLE_NAME, 
                        REFERENCED_COLUMN_NAME
                    FROM
                        INFORMATION_SCHEMA.KEY_COLUMN_USAGE
                    WHERE
                        TABLE_NAME = %s
                        AND REFERENCED_TABLE_NAME IS NOT NULL;
                """, (table,))
                fks = cursor.fetchall()
                schema["tables"][table]["foreign_keys"] = [
                    {"column": fk[0], "ref_table": fk[1], "ref_column": fk[2]}
                    for fk in fks
                ]
            except:
                pass
        
        return schema
    
    except Error as e:
        return {"tables": {}, "error": f"Error fetching schema: {e}"}
    finally:
        cursor.close()
        conn.close()

def pdf_read(pdf_paths):
    """Read text from PDF files."""
    text = ""
    try:
        for pdf_path in pdf_paths:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() or ""
        return text
    except Exception as e:
        print(f"PDF read error: {e}")
        return ""

def get_chunks(text):
    """Split text into chunks for vector store."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return text_splitter.split_text(text)

def vector_store(text_chunks):
    """Create and save FAISS vector store."""
    try:
        vector_store = FAISS.from_texts(text_chunks, embedding=get_embeddings())
        vector_store.save_local(FAISS_PATH)
        return True
    except Exception as e:
        print(f"Vector store error: {e}")
        return False

def extract_product_name_llm(user_query, chat_history, session_id):
    """Extract product name from user query using an LLM, falling back to the last product in chat history."""
    session_state = get_session_state(session_id)
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
        max_tokens=100
    )
    query_type = session_state.query
    if query_type == "continue" :
        # Format chat history
        formatted_history = "".join(
            f"Human: {m.content}\n" if isinstance(m, HumanMessage) 
            else f"AI: {m.content}\n" for m in chat_history[-5:]
        )
    else : 
        formatted_history = []

    # Define prompt for LLM
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are a precise product name extractor for a retail assistant. Extract the product name from the user query. If no product name is found in the query, strictly, identify the only last product name mentioned in the {formatted_history}. 

        Rules:
        - Valid product categories are: Electronics (TV, Laptops), Apparel (Men Clothing, Women Clothing, Unisex Clothing), Footwear (Athletic, Casual, Formal).
        - Popular brands include: Apple, Dell, HP, Lenovo, Xiaomi, Samsung.
        - If a brand is mentioned with a product (e.g., "Samsung Crystal UHD"), include the brand in the product name.
        - For vague queries (e.g., "laptop"), do NOT return a generic term; return None unless a specific product is mentioned.
        - If no product is found in the query, extract the most recent product name from the chat history (AI or Human messages).
        - CRITICAL : NEVER genearte the response with AI or Human simply pass product name like product_name.eg if you retrieved the product as Dell XPS 15 than simply return Dell XPS 15 nothing else.
        - Return only the product name as a string, or 'None' if no product is found.
        - Do NOT invent product names or use external knowledge.

        Chat History:
        {formatted_history}

        User Query: {user_query}"""),
        ("human", "{input}")
    ])

    try:
        # Format prompt and invoke LLM
        formatted_prompt = prompt.format(input=user_query)
        response = llm.invoke(formatted_prompt)
        product_name = response.content.strip()

        # Handle 'None' response
        if product_name.lower() == 'none':
            product_name = None
        else:
            # Update session state
            session_state.product_context = product_name

        return product_name

    except Exception as e:
        print(f"Error extracting product name with LLM: {e}")
        return None

def generate_sql_query(session_id, user_query, chat_history, order_id=None, intent=None,product_name=None):
    """Generate SQL query based on user query."""

    session_state = get_session_state(session_id)
    query_type = session_state.query
    if query_type == "continue":
        formatted_history = "".join(
            f"Human: {m.content}\n" if isinstance(m, HumanMessage) 
            else f"AI: {m.content}\n" for m in chat_history[-5:]
        )
    else :
        formatted_history = []
    print(formatted_history)
    session_state = get_session_state(session_id)
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
        max_tokens=300
    )
    
    schema = get_db_schema(session_id)
    if "error" in schema:
        return schema["error"], None, None
    
    category_subcategory_map = {
        "Electronics": ["TV", "Laptops"],
        "Apparel": ["Men Clothing", "Women Clothing", "Unisex Clothing"],
        "Footwear": ["Athletic", "Casual", "Formal"]
    }
    
    product_table = None
    order_table = None
    category_col = None
    sub_category_col = None
    for table, details in schema["tables"].items():
        if any(col["name"] in ["sku", "product_name", "price"] for col in details["columns"]):
            print(f"Product table found: {table}")
            product_table = table
            for col in details["columns"]:
                if col["name"].lower() == "category":
                    category_col = col["name"]
                if col["name"].lower() == "sub_category":
                    sub_category_col = col["name"]
        if any(col["name"] in ["order_id", "customer_id", "order_status"] for col in details["columns"]):
            order_table = table
    
    if not product_table and intent not in ["order_inquiry", "category_inquiry"]:
        return "No product table found in the schema.", None, None
    if not order_table and intent == "order_inquiry":
        return "No order table found in the schema.", None, None
    if not sub_category_col and intent not in ["order_inquiry", "category_inquiry"]:
        return "No sub_category column found in the product table.", None, None
    
    input_to_category_map = {
        "tv": ("Electronics", "TV"),
        "television": ("Electronics", "TV"),
        "laptop": ("Electronics", "Laptops"),
        "laptops": ("Electronics", "Laptops"),
        "computer": ("Electronics", "Laptops"),
        "apparel": ("Apparel", "Unisex Clothing"),
        "men": ("Apparel", "Men Clothing"),
        "unisex": ("Apparel", "Unisex Clothing"),
        "women": ("Apparel", "Women Clothing"),
        "shoe": ("Footwear", "Athletic"),
        "shoes": ("Footwear", "Athletic"),
        "sports shoe": ("Footwear", "Athletic"),
        "sports shoes": ("Footwear", "Athletic"),
        "sports": ("Footwear", "Athletic"),
        "casual": ("Footwear", "Casual"),
        "formal": ("Footwear", "Formal"),
        "footwear": ("Footwear", "Athletic")
    }
    
    product_category = None
    product_sub_category = None
    budget_constraint = None
    
    # Extract category and sub_category from user query
    query_lower = user_query.lower()
    for user_term, (cat, sub_cat) in input_to_category_map.items():
        if user_term in query_lower:
            product_category = cat
            product_sub_category = sub_cat
            session_state.product_context = sub_cat
            break
    
    # Extract budget constraints from user query
    budget_match = re.search(r"under\s+\$?(\d+)", query_lower)
    if budget_match:
        budget_constraint = int(budget_match.group(1))
    else:
        price_mentions = re.findall(r'(\$\d+|\d+\s+dollars|\d+\s+bucks)', query_lower)
        if price_mentions:
            for mention in price_mentions:
                if '$' in mention:
                    try:
                        budget_constraint = int(mention.replace('$', '').strip())
                        break
                    except:
                        pass
                elif 'dollars' in mention or 'bucks' in mention:
                    try:
                        budget_constraint = int(mention.split()[0].strip())
                        break
                    except:
                        pass
    
    # Handle footwear-specific keywords
    footwear_keywords = ["shoe", "shoes", "footwear", "sports shoe", "sports shoes", "sports"]
    # Only assign sub-category when explicitly stated
    if any(keyword in query_lower for keyword in footwear_keywords):
        product_category = "Footwear"
        if "formal" in query_lower:
            product_sub_category = "Formal"
        elif "casual" in query_lower:
            product_sub_category = "Casual"
        elif "athletic" in query_lower or "sports" in query_lower:
            product_sub_category = "Athletic"
        else:
            product_sub_category = None  # leave it out if unspecified

        session_state.product_context = product_sub_category
    
    # Normalize sub_category
    if product_sub_category:
        valid_subcategories = []
        for cat, subcats in category_subcategory_map.items():
            if product_category == cat:
                valid_subcategories.extend(subcats)
        
        if product_sub_category not in valid_subcategories:
            corrections = {
                "Shoe": "Athletic",
                "Shoes": "Athletic",
                "Laptop": "Laptops",
                "Men": "Men Clothing",
                "Women": "Women Clothing",
                "Unisex": "Unisex Clothing"
            }
            product_sub_category = corrections.get(product_sub_category, product_sub_category)
            
            if product_sub_category not in valid_subcategories and product_category in category_subcategory_map:
                product_sub_category = category_subcategory_map[product_category][0]
    
    if product_sub_category:
        session_state.product_context = product_sub_category
    
    context = []
    if product_category:
        context.append(f"Product category: {product_category}")
    if product_sub_category:
        context.append(f"Product sub_category: {product_sub_category}")
    if budget_constraint:
        context.append(f"Budget constraint: under ${budget_constraint}")
    if order_id:
        context.append(f"Order ID: {order_id}")
    
    context_str = "\n".join(context)

    print(f"TABLE : {product_table}")

    
    # Build query constraints
    query_constraints = ""
    if intent == "category_inquiry":
        if "electronics" in user_query.lower():
            query_constraints = f"""
            SELECT DISTINCT category, sub_category 
            FROM {product_table} 
            WHERE category = 'Electronics' AND category IS NOT NULL AND sub_category IS NOT NULL 
            ORDER BY category, sub_category;
            """
        elif "apparel" in user_query.lower():
            query_constraints = f"""
            SELECT DISTINCT category, sub_category 
            FROM {product_table} 
            WHERE category = 'Apparel' AND category IS NOT NULL AND sub_category IS NOT NULL 
            ORDER BY category, sub_category;
            """
        elif "footwear" in user_query.lower():
            query_constraints = f"""
            SELECT DISTINCT category, sub_category 
            FROM {product_table} 
            WHERE category = 'Footwear' AND category IS NOT NULL AND sub_category IS NOT NULL 
            ORDER BY category, sub_category;
            """
        else:
            query_constraints = f"""
            SELECT DISTINCT category, sub_category 
            FROM {product_table} 
            WHERE category IS NOT NULL AND sub_category IS NOT NULL 
            ORDER BY category, sub_category;
            """

    elif intent == "return_inquiry":
        print(f"PRO : {product_name}")
        if not product_name:
            return "", None, None
        else:
            return f"SELECT return_period FROM Products WHERE product_name = '{product_name}';",None,None
        
    elif intent == "warranty_inquiry" or intent == "warranty_date":
        print(f"PRO : {product_name}")
        if not product_name:
            return "", None, None
        else:
            return f"SELECT warranty_period FROM Products WHERE product_name = '{product_name}';",None,None
        
    
    elif intent == "best":
        if not product_category or not product_sub_category:
            return "", None, None
        query_constraints = f"SELECT product_name, price, description, specifications FROM {product_table}"
        if product_category and category_col:
            query_constraints += f" WHERE {category_col} = '{product_category}'"
        if product_sub_category:
            query_constraints += f" AND {sub_category_col} = '{product_sub_category}'"
        if budget_constraint:
            query_constraints += f" AND price <= {budget_constraint}"
        query_constraints += " ORDER BY price ASC LIMIT 1;"
    elif intent == "product_inquiry":
        query_constraints = f"SELECT product_name, price, description, specifications FROM {product_table}"
        if product_category and category_col:
            query_constraints += f" WHERE {category_col} = '{product_category}'"
        if product_sub_category:
            query_constraints += f" AND {sub_category_col} = '{product_sub_category}'"
        if budget_constraint:
            query_constraints += f" AND price <= {budget_constraint}"
        query_constraints += " LIMIT 5;"
    elif intent == "order_inquiry" and order_id:
        query_constraints = f"SELECT * FROM {order_table} WHERE order_id = '{order_id}';"
    elif intent == "shipping_inquiry" :
        query_constraints = f"SELECT shipping {product_table}"
        if product_category and category_col:
            query_constraints += f" WHERE {category_col} = '{product_category}'"
        if product_sub_category:
            query_constraints += f" AND {sub_category_col} = '{product_sub_category}'"
        query_constraints += " LIMIT 1;"
    elif intent != "order_inquiry":
        if product_category and category_col:
            query_constraints += f"Filter products WHERE {category_col} = '{product_category}'."
        if product_sub_category:
            query_constraints += f" Filter products WHERE {sub_category_col} = '{product_sub_category}'."
        if budget_constraint and ("budget" in query_lower or 
                                 "price" in query_lower or
                                 "under" in query_lower or
                                 "less than" in query_lower or
                                 "cheap" in query_lower or
                                 "affordable" in query_lower):
            query_constraints += f" Filter products WHERE price <= {budget_constraint}."
        if "budget" in query_lower or "affordable" in query_lower or "cheap" in query_lower:
            query_constraints += " Sort results by price ASC."
        if "count" in query_lower or "how many" in query_lower:
            query_constraints += " Use COUNT(*) to aggregate results."
        if "return" in query_lower or "warranty" in query_lower:
            query_constraints += " Include return_period or warranty_info if available in schema."
    
    schema_desc = ""
    if intent == "order_inquiry" and order_table:
        schema_desc += f"Table: {order_table} ("
        schema_desc += ", ".join([f"{col['name']} {col['data_type']}{' (PK)' if col['is_primary'] else ''}" 
                                 for col in schema["tables"][order_table]["columns"]])
        schema_desc += ")\n"
        for fk in schema["tables"][order_table]["foreign_keys"]:
            schema_desc += f"  - Foreign Key: {fk['column']} references {fk['ref_table']}({fk['ref_column']})\n"
    elif product_table:
        schema_desc += f"Table: {product_table} ("
        schema_desc += ", ".join([f"{col['name']} {col['data_type']}{' (PK)' if col['is_primary'] else ''}" 
                                 for col in schema["tables"][product_table]["columns"]])
        schema_desc += ")\n"
        for fk in schema["tables"][product_table]["foreign_keys"]:
            schema_desc += f"  - Foreign Key: {fk['column']} references {fk['ref_table']}({fk['ref_column']})\n"

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are an expert SQL query generator for MySQL. Generate a valid SQL query based on the user question and the provided database schema. Return only the SQL query as plain text, no explanations or markdown. Intent for generating this SQL query is {intent}
        

        Schema:
        {schema_desc}

        Rules:
        - Use appropriate JOINs when querying multiple tables.
        - NEVER generate the query for all the products of the database.
        - VERY CRITICAL : Consider the last product from the chat history {formatted_history} if there is history and user doesn't mention it.for eg. this is the chat history
Human: I wan tto know about products
AI: Hi there! Looking to explore some products? Here are some popular categories you might be interested in:
- Electronics: TVs, Laptops
- Apparel: Men Clothing, Women Clothing, Unisex Clothing
- Footwear: Athletic, Casual, Formal

Feel free to let me know if any specific category catches your eye, and I can provide more details or recommendations!
Human: I want to know about laptops
AI: We currently have **9** products under **Sub Category: laptops**.

**Here are the details:**

1.
**Product Name**: Apple MacBook Pro 14
**Price**: 1999.99

**Description**: 14.2-inch Professional Laptop

2.
**Product Name**: Apple MacBook Air
**Price**: 1199.99

**Description**: 13.6-inch Ultraportable Laptop

3.
**Product Name**: Dell XPS 15
**Price**: 1899.99

**Description**: 15.6-inch Premium Laptop

4.
**Product Name**: Dell Inspiron 15
**Price**: 749.99

**Description**: 15.6-inch Everyday Laptop

5.
**Product Name**: Dell Alienware m15 R7
**Price**: 2299.99

**Description**: 15.6-inch Gaming Laptop
Human: I want to know about Dell XPS 15
AI: ### Dell XPS 15
**Price:** $1899.99
**Description:** 15.6-inch Premium Laptop
**Specifications:** Processor: Intel Core i7-12700H, RAM: 16GB DDR5, Storage: 1TB SSD, Display: 15.6" 4K OLED Touch
Human: Can I get the TV less than $500
AI: ### Xiaomi Mi TV P1
**Price:** $349.99
**Description:** 43-inch 4K Smart TV
**Specifications:** Screen Size: 43", Resolution: 4K UHD, Features: Android TV, Chromecast Built-in
Human: I want to know the return policy
AI: For product, our return policy allows returns within **30 days** from the purchase date, provided the item is unused and in its original packaging.Than consider latest product mentioned as the product.
        - Clothes both can wear goes in the category for unisex clothes.
        - CRITICAL : NEVER generate this query SELECT * FROM Products LIMIT 5; whenever {intent} is 'category_inquiry'.
        - CRITICAL: Category refers to the main classification of products, such as Apparel, Electronics, or Footwear.
        - CRITICAL: Don't assume products and don't include any products outside the SQL.
        - CRITICAL: Don't include price constraints if not asked in {user_query} or not related to budget-friendly queries.
        - CRITICAL: Sub-category refers to specific types within a category, such as TV, Laptops, Men Clothing, Women Clothing, Unisex Clothing, Athletic, Casual, or Formal.
        - EXTREMELY IMPORTANT: TV and Laptops are sub-categories of Electronics, Athletic/Casual/Formal are sub-categories of Footwear, and Men Clothing/Women Clothing/Unisex Clothing are sub-categories of Apparel.
        - CRITICAL: "Shoe" or "Shoes" is NEVER a valid sub-category. For footwear, ALWAYS use Athletic, Casual, or Formal.
        - CRITICAL: Valid sub-categories are ONLY: Men Clothing, Unisex Clothing, Women Clothing, Laptops, Athletic, Formal, Casual, TV.
        - Consider price filters only when pricing or budget is mentioned.
        - For return_inquiry, consider return_period from {product_table}.If you get the product in {formatted_history} than generate the return_inquiry query for that particular product.NEVER consider the random products only consider last mentioned product from {formatted_history}
        - CRITICAL : For shipping_inquiry, consider shipping from {product_table}.If you get the product in {formatted_history} than generate the shipping query for that particular product.
        - If product name is found in the {user_query} and  {intent} is 'product_inquiry' than I want this kind of query SELECT * FROM Products where product_name="";
        -For warranty_inquiry, consider warranty_period from {product_table}.If you get the product in {formatted_history} than generate the warranty_inquiry query for that particular product.
        - For order_inquiry consider order_id from {formatted_history} if not mentioned in query.
        - For return_date, consider return_period from {product_table} and never use created_at as the date.Check the {user_query} if the product is found in the query than don't consider . NEVER consider the random products only consider last mentioned product from {formatted_history}
        - For product_inquiry, if there is no specific {product_category} or {product_sub_category} or product name in the {user_query} than I need to generate distint category and sub_catagory from {product_table}.Also generate the distint query for if user asks for the details of all products without mentioning any specific products or category consider as a vague and generate distinct query for the same.consider this product_name, price, description, specifications in genearating the SQL query for general {product_sub_category}.NEVER consider installation services and shipping details until asked
        - For product_inquiry, use {product_table} for details, inventory, pricing, etc.If brand name mentioned in the {user_query} than consider that in the genearted query.The popular brand names are Apple,Dell,HP Lenovo,Xiaomi.If someone ask like eg : I want the details about Samsung Crystal UHD AU8000 than consider as the product name with samsung including and also samsung brand.LIMIT the products to 5 like LIMIT 5;
        - For installation_inquiry, consider installation services from {product_table}.consider the product from {formatted_history}.
        - If date-specific, use {current_time}.
        - Ensure the query is safe, syntactically correct, and schema-compliant.
        - For 'best' intent, use products from {user_query}. If none specified, return an empty string.
        - For order inquiries, query '{order_table}' and filter by 'order_id' if provided.
        - For product inquiries, query '{product_table}'.And also consider the product from the {formatted_history}
        - CRITICAL : If {user_query} says that about getting more information than consider the product from {formatted_history} and provide the information about it.NEVER generate SELECT * from Products LIMIT 5; or query on whole products table.
        - CRITICAL : Always use LIMIT 5 except for 'category_inquiry'.
        - If no relevant query can be formed, return an empty string.
        - CRITICAL: Don't apply price constraints unnecessarily.
        - CRITICAL: For 'best' intent, ensure the query respects the user-specified category/sub-category from {user_query}. If none specified, return an empty string.
        {query_constraints}

        Extracted Context:
        {context_str}

        User Question: {user_query}"""),
        ("human", "{input}")
    ])
    
    try:
        formatted_prompt = prompt.format(input=user_query)
        response = llm.invoke(formatted_prompt)
        query = response.content.strip()
        
        if "sub_category = 'Shoe'" in query or "sub_category='Shoe'" in query or "sub_category=\"Shoe\"" in query:
            if product_sub_category and product_sub_category in ["Athletic", "Casual", "Formal"]:
                query = query.replace("sub_category = 'Shoe'", f"sub_category = '{product_sub_category}'")
                query = query.replace("sub_category='Shoe'", f"sub_category='{product_sub_category}'")
                query = query.replace("sub_category=\"Shoe\"", f"sub_category=\"{product_sub_category}\"")
            else:
                query = query.replace("sub_category = 'Shoe'", "sub_category = 'Athletic'")
                query = query.replace("sub_category='Shoe'", "sub_category='Athletic'")
                query = query.replace("sub_category=\"Shoe\"", "sub_category=\"Athletic\"")
        
        print(f"Generated SQL: {query}")
        return query, product_category, budget_constraint
    except Exception as e:
        print(f"Error generating SQL query: {e}")
        return f"Error generating SQL query: {e}", product_category, budget_constraint

def query_sql_database(session_id, user_query, chat_history, order_id=None, intent=None,product_name =None):
    """Execute SQL query and return results."""
    conn = get_db_connection(session_id)
    if not conn:
        return "Failed to connect to database.", None, None, None
    
    cursor = conn.cursor(dictionary=True)
    try:
        sql_query, product_category, budget_constraint = generate_sql_query(session_id, user_query, chat_history, order_id, intent,product_name)
        if not sql_query or sql_query.startswith("Error"):
            return f"Unable to process query: {sql_query}", None, None, None
        
        print(f"Executing SQL: {sql_query}")
        
        session_state = get_session_state(session_id)
        if product_category:
            session_state.product_context = product_category
        
        cursor.execute(sql_query)
        results = cursor.fetchall()
        print(f"Results : {results}")
        
        if not results:
            return "Unfortunately, there are no products listed under that selection at the moment.", sql_query, product_category, budget_constraint
        
        session_state.last_sql_results = results
        session_state.last_sql_query = sql_query
        
        return results, sql_query, product_category, budget_constraint
    
    except Error as e:
        print(f"SQL execution error: {e}")
        return f"SQL execution error: {e}", None, None, None
    finally:
        cursor.close()
        conn.close()

def detect_intent(session_id, question, chat_history):
    """Detect the intent of the user's question."""
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    
    formatted_history = "".join(
        f"Human: {m.content}\n" if isinstance(m, HumanMessage) 
        else f"AI: {m.content}\n" for m in chat_history[-5:]
    )

    input_string = f"""
You are a strict intent and context classifier for a retail virtual assistant.
 
    Your task is to analyze the user's message and determine:
    1. The user's intent
    2. Whether the message is a new query or a continuation of the previous conversation
 
 
    You must always return your result in this exact JSON format:
    {{
    "intent": "intent_label",
    "query_type": "new_query"  // or "continue"
    }}
 
    INTENT LABELS:
    - greeting  
    - capability_inquiry  
    - product_inquiry  
    - order_inquiry  
    - return_inquiry  
    - warranty_inquiry  
    - general_inquiry  
    - best  
    - return_date  
    - warranty_date  
 
    ---
 
    RULES:
 
    Intent classification:
    - "hi", "hello" → `greeting` else goodbye, how are you, → `general_inquiry`
    - If the user asks what you can do → `capability_inquiry`
    - If the user asks for a specific product, brand, or sub-category (e.g., "Samsung TV", "laptop") → `product_inquiry`
    - If the query is vague like “give me products”, “what do you sell”, “show me items” → `general_inquiry`
    - If it's about an order or delivery status/tracking → `order_inquiry`
    - If it's about a return period and includes a date → `return_date`  
    - If it's about a return period without a date → `return_inquiry`
    - If it's about warranty period and includes a date → `warranty_date`
    - If it's about warranty period without a date → `warranty_inquiry`
    - If it's asking for the best product → `best`
    - CRITICAL : Do NOT classify as `product_inquiry` if the user is:
        - Comparing brands or technologies (e.g., “What’s better, LG or Samsung?”)
        - Asking general or abstract questions about products
        → In such cases, classify as `general_inquiry` (or `general_inquiry` if unclear)
 
    Only classify queries as in-scope if they relate directly to:
    - Exploring products by brand, subcategory, or price
    - Getting product specifications or pricing
    - Understanding return or warranty period
    - Checking return eligibility using purchase date
    - Tracking order status using an order ID
 
    If a query is too vague, too complex, or outside the assistant’s supported capabilities — even if retail-related — classify it as `general_inquiry`
 
    Query type classification:
    - `new_query` → The message is fully understandable on its own and does not refer to earlier messages
    - `continue` → The message relies on a previous message or refers to a previous product/order (e.g., "Can I return it?", "Does it have warranty?", "That one")
 
    NEVER guess.
    NEVER invent a product, order, or date.
    ALWAYS follow these rules strictly.
    ONLY return the JSON. Do not include any explanation or notes.
 
    
 
    
    Chat history:
    {formatted_history}
 
    Current user message:
    {question}
 
    ---
 
    Now classify this message by returning only the intent and query_type in JSON format like:
    {{
    "intent": "intent_label",
    "query_type": "new_query"  // or "continue"
    }}
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an intent classifier."),
        ("human", "{input}")
    ])
    
    try:
        formatted_prompt = prompt.format(input=input_string)
        response = llm.invoke(formatted_prompt)
        clean = re.sub(r"```json|```", "", response.content).strip()
        
        # Parse the cleaned JSON
        data = json.loads(clean)
        intent = data.get("intent")
        query_type = data.get("query_type")
        print(f"INTENT : {intent},{query_type}")
        return intent,query_type
    except Exception as e:
        print(f"Intent detection error: {e}")
        return "general_inquiry"


def determine_data_source(intent, question):
    """Determine the appropriate data source for the query."""
    question_lower = question.lower()
    
    if intent in ["greeting", "capability_inquiry"]:
        return "none"
    elif intent in ["product_inquiry", "installation_inquiry", "count_inquiry", "order_inquiry", "best", "return_date", "warranty_date", "category_inquiry"]:
        return "sql"
    elif intent == "shipping_inquiry":
        return "sql"
    elif intent in ["return_inquiry", "warranty_inquiry"]:
        if any(category in question_lower for category in ["tv", "laptop", "computer", "apparel", "clothing", "footwear", "shoe"]):
            return "sql"
        return "sql"
    else:
        if any(keyword in question_lower for keyword in ["order", "customer", "shipping date", "delivery", "purchase"]):
            return "sql"
        return "pdf"

def extract_order_id(session_id,query, chat_history):
    """Extract order ID from query or fallback to last found in chat history."""
    session_state = get_session_state(session_id)
    query_type = session_state.query 
    patterns = [
        r'order\s+(?:id|number|#)?\s*[:#]?\s*([a-zA-Z]*\d+[a-zA-Z\d-]*)',
        r'#\s*([a-zA-Z]*\d+[a-zA-Z\d-]*)',
        r'([A-Z]{3}-\d{4}-\d{5})',
        r'order\s+([A-Z]{3}\d{4,10})',
    ]

    # Try to extract from the current query
    for pattern in patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    # Search last 10 messages in chat history
    if query_type == "continue":
        for message in reversed(chat_history[-5:]):
            if hasattr(message, 'content'):
                for pattern in patterns:
                    match = re.search(pattern, message.content, re.IGNORECASE)
                    if match:
                        return match.group(1).strip()

    return None

def get_greeting_response():
    """Generate a random greeting response in Markdown."""
    greetings = [
        """
**Hi there! I’m AIRA** — here to help you quickly find product information, understand return and warranty policies, and track your orders.\n
**Here’s what I can assist you with:**\n
• Explore products by brand, subcategory, and price\n
• Provide product information including specifications and pricing\n
• Explain return and warranty period for any product\n
• Check return eligibility based on your purchase date\n
• Track your order status using your order ID\n
\n
\n
**We currently offer products in the following categories:**\n
• Electronics (TVs, Laptops)\n
\n
\n
**Feel free to ask whenever you're ready — I'm here to help.**

        """
    ]
    return random.choice(greetings).strip()

def get_capability_response():
    """Generate a capability response."""
    return """
**Here’s what I can assist you with:**\n
• Explore products by brand, subcategory, and price\n
• Provide product information including specifications and pricing\n
• Explain return and warranty period for any product\n
• Check return eligibility based on your purchase date\n
• Track your order status using your order ID\n\n\n
\n\n 
**We currently offer products in the following categories:**\n
• Electronics (TVs, Laptops)\n
\n
**Feel free to ask whenever you're ready — I'm here to help.**
"""

def extract_date_from_text(text):
    """Extract date from text using an LLM."""
    try:
        # Initialize LLM
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=100
        )

        # Current year for fallback
        current_year = datetime.now().year
        current_time = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')

        # Prompt to extract date
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are a precise date extractor. Given a text, identify and extract any date mentioned in it. 
            Return the date in a clear, parseable format (e.g., 'March 15, 2024' or '15 March'). 
            If no year is specified, assume the current year ({current_year}). 
            If no date is found, return 'None'. 
            Handle various formats like '15th March', 'March 15', '15/03/2024', 'next Friday', etc.
            Only return the date string or 'None', nothing else.
            
            Current time: {current_time}
            Text: {text}"""),
            ("human", "{input}")
        ])

        # Format prompt and invoke LLM
        formatted_prompt = prompt.format(input=text)
        response = llm.invoke(formatted_prompt)
        date_str = response.content.strip()
        print(f"DATE : {date_str}")

        # Handle 'None' response
        if date_str.lower() == 'None':
            return None

        # Parse the extracted date
        parsed_date = parser.parse(date_str, fuzzy=True, dayfirst=True)
        now = datetime.now()

        # If the extracted date string didn't include a year, set current year
        if not any(str(current_year - 1) <= str(i) <= str(current_year + 1) for i in range(1000, 9999) if str(i) in date_str):
            parsed_date = parsed_date.replace(year=current_year)

        print (f"Datwss : {type(parsed_date)},{type(now)}")
        if parsed_date.tzinfo is not None:
            parsed_date = parsed_date.replace(tzinfo=None)
        if parsed_date > now:
            return None

        return parsed_date

    except Exception as e:
        print(f"Error extracting date with LLM: {e}")
        return None

def format_sql_results_for_human(results, sql_query, product_name,product_category=None, intent=None, user_question=None):
    print(f"Result1: {results}, {sql_query},{user_question}")
    if not results or isinstance(results, str):
        return results

    handlers = {
        "return_inquiry": handle_return_inquiry,
        "warranty_inquiry": handle_warranty_inquiry,
        "return_date": handle_return_date,
        "warranty_date": handle_warranty_date,
        "best": handle_best_product,
        "order_inquiry": handle_order_inquiry,
        "product_inquiry": handle_product_inquiry,
        "count_inquiry": handle_count_inquiry,

    }

    if intent in {"return_date", "warranty_date"}:
        return handlers[intent](results, sql_query, product_name=product_name,user_question=user_question)
    elif intent in {"return_inquiry", "warranty_inquiry", "best", "order_inquiry", "product_inquiry", "count_inquiry"}:
        return handlers[intent](results, sql_query, product_name=product_name, product_category=product_category, user_question=user_question)

    else:
        return handle_generic_product_listing(results)



# === Intent Handlers ===

def handle_return_inquiry(results, *_,product_name,product_category=None, user_question = None):
    if results and isinstance(results, list) and "return_period" in results[0]:
        return f"{product_name} comes with a return period of **{results[0]['return_period']} days** from the purchase date, provided the item is unused and in its original packaging."
    return (
        "**Our Return Policy Overview:**\n\n"
        "* Most products can be returned within **30 days** of purchase.\n"
        "* Items must be unused, in original packaging, and include all accessories.\n"
        "* Refunds are processed within 5-7 business days after we receive the item."
    )


def handle_warranty_inquiry(results, *_ , product_name,product_category, user_question = None):
    if results and isinstance(results, list) and "warranty_period" in results[0]:
        return f"{product_name} comes with a warranty period of **{results[0]['warranty_period']} days** covering manufacturing defects. Please contact us for further details."
    return (
        "**Our Warranty Policy Overview:**\n\n"
        "* Most products come with a **1-year warranty** covering manufacturing defects.\n"
        "* Warranties are valid from the purchase date and require proof of purchase.\n"
        "* Contact our support team for warranty claims or repairs."
    )


def handle_return_date(results, *_ , product_name,user_question):
    if not results or "return_period" not in results[0]:
        return "Return period information is not available."
    print(f"QUE : {user_question}")
    purchase_date = extract_date_from_text(user_question)
    if not purchase_date:
        return f"To check return eligibility of {product_name}, please provide a valid purchase date."

    days_since = (datetime.now() - purchase_date).days
    return_period = results[0]["return_period"]
    return (
        f"Yes, you're eligible for a return {product_name}. It's been **{days_since} days** since purchase, and the return period is **{return_period} days**."
        if days_since <= return_period
        else f"Unfortunately, the return window for {product_name} has passed. It's been **{days_since} days**, and the return period was **{return_period} days**."
    )

def handle_count_inquiry(results, *_, product_name,product_category=None, user_question=None):
    if results and isinstance(results, list) and "COUNT(*)" in results[0]:
        count = results[0]["COUNT(*)"]
        category_info = f" in the **{product_category}** category" if product_category else ""
        return f"There are **{count} items**{category_info}."
    return "Sorry, I couldn't determine the item count."


def handle_warranty_date(results, *_ , product_name,user_question):
    if not results or "warranty_period" not in results[0]:
        return "Warranty information is not available."

    purchase_date = extract_date_from_text(user_question)
    if not purchase_date:
        return f"To check warranty eligibility for {product_name}, please mention the date of purchase."

    days_since = (datetime.now() - purchase_date).days
    warranty_period = results[0]["warranty_period"]
    return (
        f"Your product {product_name} is still under warranty. It's been **{days_since} days** since purchase, and the warranty period is **{warranty_period} days**."
        if days_since <= warranty_period
        else f"The warranty period for {product_name} has expired. It's been **{days_since} days**, and the warranty period was only **{warranty_period} days**."
    )


def handle_best_product(results, *_ , product_name,product_category, __):
    if not results:
        return "Sorry, I couldn't find a suitable product. Could you specify a budget or sub-category?"

    product = results[0]
    name = product.get("product_name", "Unnamed Product")
    price = product.get("price", "N/A")
    description = product.get("description", "")

    return (
        "**Based on your request, the best option is:**\n\n"
        f"- **{name}** - **${price}**" + (f" ({description})" if description else "")
    )


def handle_order_inquiry(results, *_ ,product_name,product_category=None, user_question=None):
    formatted = "**Order Details:**\n\n"
    for i, order in enumerate(results, 1):
        try:
            formatted += f" **Order ID**: {order.get('order_id', 'N/A')}\n"
            formatted += f"   - **Order Date**: {order.get('order_date').strftime('%B %d, %Y') if order.get('order_date') else 'N/A'}\n"
            for field in [
                "order_status", "item", "returnable", "in_warranty",
                "shipping_method", "tracking_number", "shipping_address",
                "billing_address", "payment_method"
            ]:
                formatted += f"   - **{field.replace('_', ' ').title()}**: {order.get(field, 'N/A')}\n"

            for field in ["subtotal", "shipping_cost", "tax", "total_amount"]:
                value = float(order.get(field, 0))
                formatted += f"   - **{field.replace('_', ' ').title()}**: ${value:,.2f}\n"

            if order.get("notes"):
                formatted += f"   - **Notes**: {order['notes']}\n"

            formatted += "\n"
        except Exception as e:
            formatted += f"   - Error formatting order: {e}\n\n"

    return formatted.strip()


def handle_product_inquiry(results, sql_query, product_name,product_category=None, user_question=None):
    if not results:
        return "No product details found."
    
    if "SELECT DISTINCT" in sql_query.upper():
        return format_distinct_fields(results)

    if len(results) == 1:
        return format_single_product(results[0])

    return handle_product_list_filtering(results, sql_query, product_name,product_category)

def format_distinct_fields(results):
    if not results:
        return "No distinct values found."

    output = "### We currently offer the product in following categories:\n\n"
    grouped = {}

    for row in results:
        cat = row.get("category", "Unknown")
        sub = row.get("sub_category", "Unknown")
        grouped.setdefault(cat, []).append(sub)

    for cat, subs in grouped.items():
        output += f"**{cat}**:\n"
        for sub in sorted(set(subs)):
            output += f"- {sub}\n"
        output += "\n"

    return output.strip()


# === Single Product Formatter ===
def format_single_product(product):
    name = product.get("product_name", "Unnamed Product")
    price = f"${product['price']:.2f}" if product.get("price") else "N/A"
    desc = product.get("description", "No description available.")
    specs = product.get("specifications", "No specifications provided.")

    return f"""### {name}
**Price:** {price}  
**Description:** {desc}  
**Specifications:** {specs}"""

# === Product List Filtering + Summary ===
def handle_product_list_filtering(results, sql_query, product_name,product_category=None):
    conn = get_db_connection("12f9e21a-17b5-4629-9c27-481266c7751e")
    if not conn:
        return "Failed to connect to the product database."

    cursor = conn.cursor(dictionary=True)

    where_clause = extract_where_clause(sql_query)

    # Try to count total products under filter
    count_text = ""
    try:
        query = f"SELECT COUNT(*) AS total FROM Products {where_clause};"
        cursor.execute(query)
        count_result = cursor.fetchone()
        count = count_result.get("total", 0)
        item_word = "product" if count == 1 else "products"
        count_text = f"We currently have **{count} {item_word}** matching your query.\n\n"
    except Exception as e:
        print(f"Count query failed: {e}")

    return count_text + handle_generic_product_listing(results,product_name)

# === Extract WHERE Clause From SQL ===
def extract_where_clause(sql_query):
    match = re.search(r'\bWHERE\b(.*?)(?:\bLIMIT\b|$)', sql_query, re.IGNORECASE | re.DOTALL)
    if match:
        return f"WHERE {match.group(1).strip()}"
    return ""

# === Format Product List ===
def handle_generic_product_listing(results, product_name=None):
    if not results:
        return "No products found."

    output = "**Here are a few you might be interested in:**\n\n"
    for i, item in enumerate(results, 1):
        parts = [f"**{i}. {item.get('product_name', 'Unnamed Product')}**"]

        # Basic product fields
        for key in [ "price" ]:
            val = item.get(key)
            if val:
                label = key.replace("_", " ").title()
                formatted_val = f"${val:.2f}" if key == "price" else val
                parts.append(f"- **{label}**: {formatted_val}")

        # Optional fields
        if item.get("description"):
            parts.append(f"- **Description**: {item['description']}")

        output += "\n".join(parts) + "\n\n"

    return output.strip()



@traceable
def get_conversational_chain(session_id, ques, chat_history):
    """Generate a conversational response based on user query and intent."""
    session_state = get_session_state(session_id)
    intent,query_type = detect_intent(session_id, ques, chat_history)
    session_state.last_intent = intent
    session_state.query = query_type

    if query_type == "new_query":
        session_state.product_context = None
        session_state.last_order_id = None
        session_state.last_sql_results = None
        session_state.last_sql_query = None
        
    if intent == "greeting":
        return get_greeting_response()
    elif intent == "capability_inquiry":
        return get_capability_response()

    data_source = determine_data_source(intent, ques)

    if data_source == "sql":
        product_name = extract_product_name_llm(ques,chat_history,session_id)
        print(f"P : {product_name}")
        if intent in ['return_inquiry', 'warranty_inquiry','warranty_date','return_date','installation_inquiry'] and not product_name:
            return "**I wasn’t able to find enough information to answer that.** \n\n Please ask your question again and include the product name so I can help you accurately."
        order_id = extract_order_id(session_id,ques,chat_history)
        if intent == "order_inquiry" and not order_id:
            return "To provide details about your order, please share your order ID."
        retrieved_data, sql_query, product_category, budget_constraint = query_sql_database(session_id, ques, chat_history, order_id, intent,product_name)
        formatted_data = format_sql_results_for_human(retrieved_data, sql_query,product_name,product_category, intent, ques)

        if intent == "best" and (not product_category or formatted_data == "No matching data found."):
            return "Could you specify which type of product you're looking for, like a TV, laptop, or footwear?"

        if formatted_data and isinstance(formatted_data, str) and formatted_data != "No relevant data found.":
            if intent in ["category_inquiry", "product_inquiry","return_inquiry","warranty_inquiry","return_date","order_inquiry","count_inquiry"]:
                return formatted_data

            llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0.7,
                api_key=os.getenv("OPENAI_API_KEY")
            )
            formatted_history = "".join(
                f"Human: {m.content}\n" if isinstance(m, HumanMessage) 
                else f"AI: {m.content}\n" for m in chat_history[-5:]
            )
            product_context = session_state.product_context if session_state.product_context else ""
            last_order_id = session_state.last_order_id if session_state.last_order_id else ""
            category_info = """
            **If you're looking to explore our products, here are some categories we currently offer:**\n
            \n
            • **Electronics** – TVs and Laptops\n
            \n
            **Let me know if you'd like to browse a specific category, brand, or price range.**

            """
            prompt = ChatPromptTemplate.from_messages([
                ("system", f"""You are a friendly Retail Assistant for an online store. Rephrase pre-formatted SQL query results into an engaging, conversational, and personalized response using Markdown formatting.

                Capabilities:
                - Order Management: Track orders, check delivery status, provide purchase details.
                - Product Information: Share details about TVs, laptops, apparel, footwear.
                - Recommendations: Suggest budget-friendly or category-specific products.
                - Policy Guidance: Explain shipping, return, refund, and warranty policies.
                - Customer Support: Answer general inquiries and assist with shopping.

                {category_info}

                Rules:
                - Rephrase SQL results to sound warm and conversational using Markdown.
                - CRITICAL : Avoid mentioning "AI", "SQL", or technical details.
                - Use user query and chat history to personalize the response.
                - If product context or order ID exists, reference it subtly.
                - For empty results, NEVER suggest alternatives.
                - NEVER suggest any product on your knowledge or chat history.
                - Keep responses concise (2-3 sentences).
                - NEVER include product id; use product name.
                - If order found, mention order date and returnable status or warranty using Markdown.
                - Never suggest products outside provided categories/sub-categories.
                - If no order ID for order_inquiry, respond with: "To provide details about your order, please share your order ID."

                Current Time: {datetime.now()}
                Recent Chat History:
                {formatted_history}
                Product Context: {product_context}
                Last Order ID: {last_order_id}
                User Query: {ques}

                **Formatted SQL Results:**
                {formatted_data}"""),
                ("human", "{input}")
            ])
            try:
                formatted_prompt = prompt.format(input=ques)
                response = llm.invoke(formatted_prompt)
                return response.content.strip()
            except Exception as e:
                print(f"Error enhancing SQL response with LLM: {e}")
                return formatted_data
        else:
            return "To provide details about your order, please share your order ID." if intent == "order_inquiry" else "I couldn't find any relevant information. Could you clarify or specify what you're looking for?"

    if intent == "general_inquiry":
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.7,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        if query_type == "continue":
            formatted_history = "".join(
                f"Human: {m.content}\n" if isinstance(m, HumanMessage) 
                else f"AI: {m.content}\n" for m in chat_history[-5:]
            )
        else : 
            formatted_history = []
        product_context = session_state.product_context if session_state.product_context else ""
        last_order_id = session_state.last_order_id if session_state.last_order_id else ""
        category_info = """
        **If you're looking to explore our products, here are some categories we currently offer:**\n
            \n
            • **Electronics** – TVs and Laptops\n
            \n
            **Let me know if you'd like to browse a specific category, brand, or price range.**
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are a friendly Retail Assistant for an online store. Respond to vague or general user queries in a natural, engaging, and conversational way using Markdown formatting. Avoid static responses. Interpret the query, consider recent chat history, and offer helpful suggestions aligned with the store's offerings.NEVER answer out of database.

            Capabilities:
            - Order Management: Track orders, check delivery status, provide purchase details.
            - Product Information: Share details about TVs, laptops, apparel, footwear.
            - Recommendations: Suggest budget-friendly or category-specific products.
            - Policy Guidance: Explain shipping, return, refund, and warranty policies.
            - Customer Support: Answer general inquiries and assist with shopping.

            {category_info}

            Rules:
            - Respond in a warm, conversational tone using Markdown.
            - CRITICAL : NEVER use emojis in the response.
            - CRITICAL : Avoid mentioning "AI", "SQL", or technical details.
            - CRITICAL : Detect if the user's message is small talk (e.g., "thanks", "how are you", "bye") and reply appropriately.
                Consider One Star Electronics in the responses of small talks
                Respond to these common small talk types:
                - Greetings (hi, hello, hey, good morning)
                - Farewells (bye, goodbye, see you later)
                - Gratitude (thanks, thank you, appreciate it)
                - Politeness (how are you, how’s it going)
            - CRITICAL : NEVER respond to that are not related to Retail or Order.e.g.what is dog or what is ai.
            - IMPORTANT:
                If the user query is not related to retail, orders, products, return/refund/warranty/shipping policies, or product availability/pricing — DO NOT answer.
                Instead, respond with:
                "Sorry I can't help you with it.  
                *Here’s what I can assist you with:*  
                • Explore products by brand, subcategory, and price  
                • Provide detailed product information including specifications and pricing  
                • Explain return and warranty policies for any product  
                • Check return eligibility based on your purchase date  
                • Track your order status using your order ID"

            - Use user query and chat history to personalize.
            - If product context or order ID exists, reference it subtly.
            - For vague queries, suggest exploring popular categories using a Markdown list.
            - If query hints at shopping intent, highlight relevant products/categories.
            - Never suggest products outside provided categories/sub-categories by using your knowledge.
            - NEVER answer out of database and don't mislead with data.
            - CRITICAL : NEVER answer using your own knowledge
            - CRITICAL : For any queries which are outside the scope of Assistant's capabilities, which are as following: 
            • Explore products by brand, subcategory, and price\n
            • Provide detailed product information including specifications and pricing\n
            • Explain return and warranty policies for any product\n
            • Check return eligibility based on your purchase date\n
            • Track your order status using your order ID\n
            consider as non-relevant, and respond with: Sorry I can't help you with it\n *Here’s what I can assist you with:**\n
            \n
            \n
            **We currently offer products in the following categories:**\n
            • Electronics (TVs, Laptops)\n
            \n
            \n
            **Feel free to ask whenever you're ready — I'm here to help
            - Keep responses concise (2-3 sentences).

            Current Time: {datetime.now()}
            Recent Chat History:
            {formatted_history}
            Product Context: {product_context}
            Last Order ID: {last_order_id}

            User Query: {ques}"""),
            ("human", "{input}")
        ])
        try:
            formatted_prompt = prompt.format(input=ques)
            response = llm.invoke(formatted_prompt)
            return response.content.strip()
        except Exception as e:
            print(f"Error generating LLM response for general_inquiry: {e}")
            return "I'm here to help with your shopping needs! Want to explore our TVs, apparel, or maybe some footwear?"

    try:
        if os.path.exists(FAISS_PATH):
            faiss_index = FAISS.load_local(FAISS_PATH, get_embeddings(), allow_dangerous_deserialization=True)
            docs = faiss_index.similarity_search(ques, k=5)
            retrieved_data = "\n".join([doc.page_content for doc in docs]) if docs else "No relevant data found."
        else:
            retrieved_data = "No product data available. Please upload PDFs with product information."
    except Exception as e:
        retrieved_data = f"Error retrieving data: {e}"

    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    formatted_history = "".join(
        f"Human: {m.content}\n" if isinstance(m, HumanMessage) 
        else f"AI: {m.content}\n" for m in chat_history[-5:]
    )
    product_context = session_state.product_context if session_state.product_context else ""
    category_info = """
    **If you're looking to explore our products, here are some categories we currently offer:**\n
    \n
    • **Electronics** – TVs and Laptops\n
    \n
    **Let me know if you'd like to browse a specific category, brand, or price range.**
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You're a friendly Retail Assistant. Handle queries about shipping, warranty, returns, refunds, products (TV, Laptops, Apparel, Footwear), and customer orders using Markdown formatting.

        Capabilities:
        - Order Management: Track orders, check delivery status, provide purchase details.
        - Product Information: Share details, specs, and features about TVs, laptops, apparel, footwear.
        - Recommendations: Suggest budget-friendly or category-specific products.
        - Policy Guidance: Explain shipping, return, refund, and warranty policies.
        - Customer Support: Answer general inquiries and assist with shopping.

        Intent: {intent}
        Data Source: pdf
        Product Context: {product_context}
        {category_info}

        Answer Rules:
        - Provide clear, concise, friendly responses using Markdown.
        - Use retrieved data to answer, focusing on relevant details.
        - If retrieved data is empty or irrelevant, respond with: "I couldn't find any relevant information. Could you clarify or specify what you're looking for?"
        - Avoid generic phrases like "Let me check for you".
        - For out-of-scope queries: "Sorry, I can only answer about our store's products and policies."
        - Always maintain a helpful, retail-oriented tone.

        Recent Chat History:
        {formatted_history}

        Question: {ques}
        **Retrieved Data:**
        {retrieved_data}"""),
        ("human", "{input}")
    ])

    try:
        formatted_prompt = prompt.format(input=ques)
        response = llm.invoke(formatted_prompt)
        return response.content.replace("AI:", "").strip()
    except Exception as e:
        print(f"Error in get_conversational_chain: {e}")
        return retrieved_data if retrieved_data and isinstance(retrieved_data, str) and retrieved_data != "No relevant data found." else "Sorry, I couldn't process your request. Could you clarify what you're looking for?"

def user_input(session_id, user_question, chat_history):
    """Process user input and generate response."""
    try:
        response = get_conversational_chain(session_id, user_question, chat_history)
        
        chat_history.append(HumanMessage(content=user_question))
        chat_history.append(AIMessage(content=response))
        
        return response
    except Exception as e:
        print(f"Error in user_input: {e}")
        return f"Sorry, something went wrong. Please try again."

def create_db_connection():
    """Create a MySQL database connection."""
    session_id = list(session_store.keys())[0] if session_store else str(uuid.uuid4())
    session_state = get_session_state(session_id)
    try:
        connection = mysql.connector.connect(
            host=session_state.db_params["host"],
            port=session_state.db_params["port"],
            database=session_state.db_params["database"],
            user=session_state.db_params["user"],
            password=session_state.db_params["password"],
            ssl_disabled=not session_state.db_params["use_ssl"]
        )
        return connection
    except Error as e:
        print(f"Database connection error: {e}")
        return None

def execute_query(connection, query, params=None, fetch=True):
    """Execute a SQL query with optional parameters."""
    cursor = connection.cursor(dictionary=True)
    try:
        cursor.execute(query, params or ())
        if fetch:
            result = cursor.fetchall()
            connection.commit()
            return result
        else:
            connection.commit()
            return cursor
    except Error as e:
        print(f"Error executing query: {e}")
        connection.rollback()
        raise e
    finally:
        cursor.close()

def get_or_create_chat_session_by_client_id(client_id):
    """Get or create a chat session for a client ID."""
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
        
        if results:
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

def save_chat_message(chat_id, role, message):
    """Save a chat message to the database."""
    connection = create_db_connection()
    if connection is None:
        return False
    
    try:
        check_query = """
            SELECT COUNT(*) as count FROM chat_messages 
            WHERE chat_id = %s AND role = %s AND message = %s 
            AND timestamp > NOW() - INTERVAL 5 SECOND
        """
        result = execute_query(connection, check_query, (chat_id, role, message), fetch=True)
        
        if result[0]['count'] > 0:
            print(f"Skipping duplicate message for chat ID: {chat_id}")
            return True
            
        query = """
            INSERT INTO chat_messages (chat_id, role, message, timestamp) 
            VALUES (%s, %s, %s, NOW())
        """
        execute_query(connection, query, (chat_id, role, message), fetch=False)
        return True
    except Exception as e:
        print(f"Error saving chat message: {e}")
        return False
    finally:
        if connection and connection.is_connected():
            connection.close()

def get_chat_history_from_db(chat_id, raw=False):
    """Retrieve chat history for a specific chat ID."""
    connection = create_db_connection()
    if connection is None:
        return [] if not raw else []
    
    try:
        query = """
            SELECT role, message, timestamp 
            FROM chat_messages 
            WHERE chat_id = %s 
            ORDER BY timestamp ASC
        """
        results = execute_query(connection, query, (chat_id,), fetch=True)
        if raw:
            return results
        return [
            HumanMessage(content=msg['message']) if msg['role'] == 'user' else AIMessage(content=msg['message'])
            for msg in results
        ]
    except Exception as e:
        print(f"Error retrieving chat history: {e}")
        return [] if not raw else []
    finally:
        if connection and connection.is_connected():
            connection.close()

def mark_chat_as_deleted(chat_id):
    """Mark a chat session as deleted."""
    connection = create_db_connection()
    if connection is None:
        return False
    
    try:
        query = "UPDATE chat_sessions SET deleted = TRUE WHERE id = %s"
        execute_query(connection, query, (chat_id,), fetch=False)
        return True
    except Exception as e:
        print(f"Error marking chat as deleted: {e}")
        return False
    finally:
        if connection and connection.is_connected():
            connection.close()

@app.route('/start_session', methods=['POST'])
def start_session():
    """Start a new session and return a session ID."""
    session_id = str(uuid.uuid4())
    session_store[session_id] = SessionState()
    return jsonify({"status": "success", "session_id": session_id})

@app.route('/configure_db', methods=['POST'])
def configure_db():
    """Configure database credentials."""
    data = request.json
    session_id = data.get('session_id')
    if not session_id:
        return jsonify({"status": "error", "message": "Session ID is required.", "error_code": "MISSING_SESSION_ID"}), 400
    
    use_env = data.get('use_env', True)
    
    session_state = get_session_state(session_id)
    if not use_env:
        if not all(key in data for key in ["host", "port", "database", "user", "password"]):
            return jsonify({"status": "error", "message": "Missing required database credentials.", "error_code": "MISSING_CREDENTIALS"}), 400
        session_state.db_params = {
            "host": data.get('host'),
            "port": data.get('port'),
            "database": data.get('database'),
            "user": data.get('user'),
            "password": data.get('password'),
            "use_ssl": data.get('use_ssl', False)
        }
    
    session_state.db_configured = test_db_connection(session_id)
    
    if session_state.db_configured:
        valid, message = validate_db_schema(session_id)
        if not valid:
            return jsonify({"status": "error", "message": message, "error_code": "INVALID_SCHEMA"}), 400
        return jsonify({"status": "success", "message": "Connected to MySQL database!"})
    else:
        return jsonify({"status": "error", "message": "Failed to connect. Please check credentials and server status.", "error_code": "DB_CONNECTION_FAILED"}), 400

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    """Upload and process PDF files."""
    session_id = request.form.get('session_id')
    if not session_id:
        return jsonify({"status": "error", "message": "Session ID is required.", "error_code": "MISSING_SESSION_ID"}), 400
    
    if 'files' not in request.files:
        return jsonify({"status": "error", "message": "No files provided.", "error_code": "NO_FILES"}), 400
    
    files = request.files.getlist('files')
    temp_paths = []
    
    try:
        for file in files:
            if file and file.filename.endswith('.pdf'):
                temp_path = os.path.join(BASE_DIR, f"temp_{uuid.uuid4()}.pdf")
                file.save(temp_path)
                temp_paths.append(temp_path)
        
        if not temp_paths:
            return jsonify({"status": "error", "message": "No valid PDF files uploaded.", "error_code": "INVALID_FILES"}), 400
        
        raw_text = pdf_read(temp_paths)
        if raw_text:
            text_chunks = get_chunks(raw_text)
            if vector_store(text_chunks):
                return jsonify({"status": "success", "message": f"Successfully processed {len(temp_paths)} PDFs into vector store!"})
            else:
                return jsonify({"status": "error", "message": "Failed to create vector store.", "error_code": "VECTOR_STORE_FAILED"}), 500
        else:
            return jsonify({"status": "error", "message": "Could not extract text from PDFs.", "error_code": "PDF_PROCESSING_FAILED"}), 500
    
    finally:
        for path in temp_paths:
            if os.path.exists(path):
                os.remove(path)

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat interactions."""
    data = request.json
    session_id = data.get('chat_id')
    user_question = data.get('question')
    client_id = data.get('client_id')
    
    if not session_id:
        return jsonify({"status": "error", "message": "Session ID is required.", "error_code": "MISSING_SESSION_ID"}), 400
    if not user_question:
        return jsonify({"status": "error", "message": "No question provided.", "error_code": "MISSING_QUESTION"}), 400
    if not client_id:
        return jsonify({"status": "error", "message": "Client ID is required.", "error_code": "MISSING_CLIENT_ID"}), 400
    
    session_state = get_session_state(session_id)
    chat_id = get_or_create_chat_session_by_client_id(client_id)
    
    if os.path.exists(FAISS_PATH) or session_state.db_configured:
        try:
            chat_history = get_chat_history_from_db(chat_id)
            save_chat_message(chat_id, 'user', user_question)
            response = user_input(session_id, user_question, chat_history)
            save_chat_message(chat_id, 'assistant', response)
            
            formatted_history = [
                {"role": "user" if isinstance(msg, HumanMessage) else "assistant", "content": msg.content}
                for msg in chat_history
            ]
            
            return jsonify({
                "status": "success",
                "session_id": session_id,
                "chat_id": chat_id,
                "response": response,
                "chat_history": formatted_history
            })
        except Exception as e:
            print(f"Error in chat endpoint: {e}")
            return jsonify({"status": "error", "message": "Internal server error.", "error_code": "INTERNAL_ERROR"}), 500
    else:
        return jsonify({
            "status": "error",
            "message": "Please configure MySQL database credentials or upload PDFs with product information to continue.",
            "error_code": "NO_DATA_SOURCE"
        }), 400

@app.route('/clear_chat', methods=['POST'])
def clear_chat():
    """Clear chat history for a session."""
    data = request.json
    session_id = data.get('chat_id')
    if not session_id:
        return jsonify({"status": "error", "message": "Session ID is required.", "error_code": "MISSING_SESSION_ID"}), 400
    
    session_state = get_session_state(session_id)
    session_state.chat_history = []
    session_state.product_context = None
    session_state.last_order_id = None
    session_state.last_sql_results = None
    session_state.last_sql_query = None
    session_state.last_intent = None
    return jsonify({"status": "success", "message": "Chat history cleared."})

@app.route('/debug_info', methods=['POST'])
def debug_info():
    """Return debug information for a session."""
    data = request.json
    session_id = data.get('session_id')
    if not session_id:
        return jsonify({"status": "error", "message": "Session ID is required.", "error_code": "MISSING_SESSION_ID"}), 400
    
    session_state = get_session_state(session_id)
    return jsonify({
        "product_context": session_state.product_context,
        "last_order_id": session_state.last_order_id,
        "last_intent": session_state.last_intent,
        "last_sql_results": session_state.last_sql_results,
        "last_sql_query": session_state.last_sql_query
    })

@app.route('/api/chat/session', methods=['POST'])
def get_chat_session():
    """Get or create a chat session for the provided client ID."""
    try:
        data = request.json
        client_id = data.get('client_id')
        
        if not client_id:
            return jsonify({
                "status": "error",
                "message": "Client ID is required"
            }), 400
            
        chat_id = get_or_create_chat_session_by_client_id(client_id)
        session_id = str(uuid.uuid4())
        session_store[session_id] = SessionState()
        
        return jsonify({
            "status": "success",
            "session_id": session_id,
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
        history = get_chat_history_from_db(chat_id, raw=True)
        formatted_history = [
            {
                "role": msg['role'],
                "content": msg['message'],
                "timestamp": msg['timestamp'].isoformat() if isinstance(msg['timestamp'], datetime) else msg['timestamp']
            }
            for msg in history
        ]
        
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
            keys_to_delete = [key for key, state in session_store.items() if hasattr(state, 'chat_id') and state.chat_id == chat_id or key == chat_id]
            for key in keys_to_delete:
                if key in session_store:
                    del session_store[key]
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

@app.route('/health', methods=['GET'])
def health():
    """Check API health and database connectivity."""
    session_id = str(uuid.uuid4())
    session_store[session_id] = SessionState()
    db_status = test_db_connection(session_id)
    del session_store[session_id]
    
    return jsonify({
        "status": "success",
        "api": "running",
        "database": "connected" if db_status else "disconnected"
    })

@app.route('/')
def home():
    return {"message": "Retail Flask API running!"}

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)