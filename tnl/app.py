import os
import pandas as pd
from flask import Flask, request, jsonify
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import uuid
from werkzeug.utils import secure_filename
from flask_cors import CORS
import mysql.connector
from mysql.connector import Error
from langchain_community.utilities import sql_database
from langchain_core.messages import AIMessage, HumanMessage
import json
import logging
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timedelta
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    logger.error("OPENAI_API_KEY not found in environment variables")
    raise ValueError("OPENAI_API_KEY is required")

CORS(app)

# Directory configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "Uploads")
FAISS_PATH = os.path.join(BASE_DIR, "faiss_index")
for directory in [UPLOAD_FOLDER, FAISS_PATH]:
    os.makedirs(directory, exist_ok=True)

# Database configuration
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "127.0.0.1"),
    "port": int(os.getenv("DB_PORT", "3306")),
    "database": os.getenv("DB_NAME", "csv_chatbot"),
    "user": os.getenv("DB_USER", ""),
    "password": os.getenv("DB_PASSWORD", "")
}

MYSQL_QUERY_CONFIG = {
    "host": os.getenv("MYSQL_QUERY_HOST", "127.0.0.1"),
    "port": int(os.getenv("MYSQL_QUERY_PORT", "3306")),
    "database": os.getenv("MYSQL_QUERY_NAME", ""),
    "user": os.getenv("MYSQL_QUERY_USER", "root"),
    "password": os.getenv("MYSQL_QUERY_PASSWORD", "Karna!21")}

# Initialize LLM
try:
    llm = ChatOpenAI(model="gpt-4o", api_key=openai_api_key)
except Exception as e:
    logger.error(f"Failed to initialize LLM: {str(e)}")
    raise

# Global vector store
vector_store: Optional[FAISS] = None

# Session context cache
session_context_cache = {}

def get_db_connection(config: Dict[str, Any] = DB_CONFIG) -> Optional[mysql.connector.connection.MySQLConnection]:
    """Establish a database connection."""
    try:
        conn = mysql.connector.connect(
            host=config["host"],
            port=config["port"],
            database=config["database"],
            user=config["user"],
            password=config["password"],
            connection_timeout=10
        )
        logger.debug(f"Database connection established to {config['host']}")
        return conn
    except Error as e:
        logger.error(f"Database connection error: {e}")
        return None

def execute_query(connection: mysql.connector.connection.MySQLConnection, 
                 query: str, 
                 params: tuple = None, 
                 fetch: bool = True) -> Any:
    """Execute a SQL query."""
    cursor = None
    try:
        cursor = connection.cursor(dictionary=True)
        cursor.execute(query, params or ())
        if fetch:
            result = cursor.fetchall()
        else:
            result = cursor
        connection.commit()
        return result
    except Error as e:
        logger.error(f"Error executing query: {e}")
        connection.rollback()
        raise
    finally:
        if cursor:
            cursor.close()

def create_session() -> str:
    """Create a new chat session."""
    session_id = str(uuid.uuid4())
    conn = get_db_connection()
    if not conn:
        raise Exception("Database connection failed")
    
    try:
        query = """
            INSERT INTO chat_sessions (id, created_at, deleted)
            VALUES (%s, %s, %s)
        """
        execute_query(conn, query, (session_id, datetime.now(), False), fetch=False)
        logger.info(f"Created new session: {session_id}")
        return session_id
    finally:
        if conn and conn.is_connected():
            conn.close()

def save_chat_message(session_id: str, role: str, message: str) -> bool:
    """Save a chat message to the database."""
    conn = get_db_connection()
    if not conn:
        logger.error("Failed to save chat message: No database connection")
        return False
    
    try:
        message_id = str(uuid.uuid4())
        query = """
            INSERT INTO chat_messages (id, chat_id, role, message, timestamp)
            VALUES (%s, %s, %s, %s, %s)
        """
        execute_query(conn, query, (message_id, session_id, role, message, datetime.now()), fetch=False)
        logger.debug(f"Saved message for session {session_id}")
        return True
    except Exception as e:
        logger.error(f"Error saving chat message: {e}")
        return False
    finally:
        if conn and conn.is_connected():
            conn.close()

def mark_session_as_deleted(session_id: str) -> bool:
    """Mark a session as deleted."""
    conn = get_db_connection()
    if not conn:
        logger.error("Failed to mark session as deleted: No database connection")
        return False
    
    try:
        query = "UPDATE chat_sessions SET deleted = TRUE, last_order_id = NULL WHERE id = %s"
        execute_query(conn, query, (session_id,), fetch=False)
        logger.info(f"Marked session {session_id} as deleted")
        
        if session_id in session_context_cache:
            del session_context_cache[session_id]
            
        return True
    except Exception as e:
        logger.error(f"Error marking session as deleted: {e}")
        return False
    finally:
        if conn and conn.is_connected():
            conn.close()

def retrieve_chat_history(session_id: str) -> Dict[str, Any]:
    """Retrieve chat history and context for a session."""
    conn = get_db_connection()
    if not conn:
        logger.error("Database connection failed for chat history")
        raise Exception("Database connection failed")
    
    try:
        query = """
            SELECT cm.role, cm.message, cm.timestamp
            FROM chat_messages cm
            JOIN chat_sessions cs ON cm.chat_id = cs.id
            WHERE cm.chat_id = %s AND cs.deleted = FALSE
            ORDER BY cm.timestamp DESC
        """
        messages = execute_query(conn, query, (session_id,), fetch=True)
        
        if not messages and not execute_query(conn, 
            "SELECT id FROM chat_sessions WHERE id = %s AND deleted = FALSE", 
            (session_id,), fetch=True):
            logger.warning(f"Session not found: {session_id}")
            raise Exception("Session not found or deleted")
        
        formatted_messages = [
            HumanMessage(content=msg["message"]) if msg["role"] == "user"
            else AIMessage(content=msg["message"])
            for msg in reversed(messages)  # Reverse to maintain chronological order
        ]
        
        if session_id not in session_context_cache:
            session_context_cache[session_id] = {
                "order_ids": set(),
                "last_order_id": None,
                "email": None,
                "last_query_time": datetime.now(),
                "last_intent": None
            }
        
        logger.info(f"Retrieved chat history for session {session_id}")
        return {
            "messages": formatted_messages,
            "order_ids": session_context_cache[session_id]["order_ids"],
            "last_order_id": session_context_cache[session_id]["last_order_id"],
            "email": session_context_cache[session_id]["email"],
            "last_intent": session_context_cache[session_id]["last_intent"]
        }
    except Exception as e:
        logger.error(f"Chat history retrieval error: {e}")
        raise
    finally:
        if conn and conn.is_connected():
            conn.close()


PROMPT_TEMPLATE = ChatPromptTemplate.from_template(
    """
    You are an assistant for a Transport & Logistics company. Your task is to extract an order ID from the current query or Session order Id. Order IDs are typically in the format 'ORD' followed by numbers (e.g., ORD123).
    Current Query: {query}
    Session Order Id: {order_id}

    Instructions:
    - Check the current query first for an order ID.
    - CRITICAL: If not found, check the Session Order Id for the latest mentioned order ID.
    - CRITICAL: If no Order id is found in Current Query or Session Order Id, pass empty strings.
    - Return only the order ID as a string, or an empty string if no order ID is found.
    - NEVER assume the order ID.
    - Do not invent or assume any order IDs.
    - CRITICAL: Just simply pass order id in response, nothing else like don't assign it to the variable.
    - Response must be <order_id> or ""
    """
)

def format_chat_history_and_extract_order_id(session_id: str, query: str) -> Tuple[str, str]:
    """Format chat history and extract order ID using LLM."""
    try:
        # Retrieve chat history
        context = retrieve_chat_history(session_id)
        messages = context["messages"]

        # Format the last 5 messages
        formatted_history = ""
        for msg in messages[-5:]:
            role = "Human" if isinstance(msg, HumanMessage) else "AI"
            formatted_history += f"{role}: {msg.content}\n"

        # Retrieve last order ID from database
        last_order_id = ""
        with get_db_connection() as conn:
            sql_query = """
                SELECT last_order_id FROM chat_sessions WHERE id = %s;
            """
            result = execute_query(conn, sql_query, (session_id,), fetch=True)
            print(f"EXE : {result}")
            if result:
                last_order_id = result[0]['last_order_id'] or ""

        # Extract order ID using LLM
        final_prompt = PROMPT_TEMPLATE.format(query=query, order_id=last_order_id)
        response = llm.invoke(final_prompt)
        order_id = response.content.strip()

        # Validate order ID
        return formatted_history, order_id if order_id.startswith("ORD") else ""
    except Exception as e:
        logger.error(f"Error formatting history or extracting order ID: {e}")
        return "", ""

def process_csv(file_path: str) -> Optional[FAISS]:
    """Process CSV file and create FAISS vector store."""
    global vector_store
    try:
        df = pd.read_csv(file_path)
        csv_text = df.to_string(index=False)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(csv_text)
        
        metadatas = [{"chunk_id": str(uuid.uuid4()), "source": file_path, "index": i} 
                     for i in range(len(chunks))]
        
        embeddings = OpenAIEmbeddings(api_key=openai_api_key)
        vector_store = FAISS.from_texts(chunks, embeddings, metadatas=metadatas)
        vector_store.save_local(FAISS_PATH)
        logger.info(f"Processed CSV and saved FAISS index: {file_path}")
        return vector_store
    except Exception as e:
        logger.error(f"Error processing CSV: {e}")
        raise

def update_session_context(session_id: str, intent: str, query: str, order_id: Optional[str] = None, email: Optional[str] = None, waiting_for: Optional[str] = None) -> None:
    """Update the session context."""
    if session_id not in session_context_cache:
        session_context_cache[session_id] = {
            "order_ids": set(),
            "last_order_id": None,
            "email": None,
            "last_query_time": datetime.now(),
            "last_intent": None,
            "waiting_for": None
        }
    
    context = session_context_cache[session_id]
    
    if order_id:
        context["order_ids"].add(order_id)
        context["last_order_id"] = order_id
    
    if email:
        context["email"] = email
    else:
        prompt_template = ChatPromptTemplate.from_template(
            """
            Extract an email address from the following query if present.
            Query: {query}
            Return the email address as a string, or an empty string if none is found.
            """
        )
        final_prompt = prompt_template.format(query=query)
        response = llm.invoke(final_prompt)
        email = response.content.strip()
        if email and "@" in email:
            context["email"] = email
    
    context["last_query_time"] = datetime.now()
    context["last_intent"] = intent
    context["waiting_for"] = waiting_for
    
    if context["last_order_id"]:
        conn = get_db_connection()
        if conn:
            try:
                query = "UPDATE chat_sessions SET last_order_id = %s WHERE id = %s"
                execute_query(conn, query, (context["last_order_id"], session_id), fetch=False)
                logger.debug(f"Updated last_order_id to {context['last_order_id']} for session {session_id}")
            except Exception as e:
                logger.error(f"Error updating last_order_id: {e}")
            finally:
                if conn and conn.is_connected():
                    conn.close()

def clean_old_contexts() -> None:
    """Clean up old session contexts."""
    now = datetime.now()
    expired_sessions = [
        session_id for session_id, context in session_context_cache.items()
        if now - context["last_query_time"] > timedelta(hours=2)
    ]
    
    for session_id in expired_sessions:
        del session_context_cache[session_id]
        logger.debug(f"Cleaned expired session context: {session_id}")

def is_logistics_query(query: str) -> bool:
    """Check if the query is related to transport, logistics, or orders."""
    prompt_template = ChatPromptTemplate.from_template(
        """
        Determine if the following query is related to transport, logistics, or order-related topics.
        Examples of relevant topics include tracking shipments, rescheduling deliveries, changing addresses,
        downloading invoices, reporting order issues, or general logistics FAQs,Hi hello.
        The below mentioned always goes in the relevant queries.

        csv: General FAQs about ordering, payments, shipping, delivery, warranties, returns, dealers, or technical support.
        - mysql: Queries about specific customer data (e.g., order status, delivery dates, invoices) or actions requiring an order ID or email.
        - reschedule_delivery: Requests to change delivery dates or times.
        - address_change: Requests to update delivery addresses.
        - general: Greetings (e.g., "Hi") or unrelated queries.
        - capabilities: Questions about the assistant's capabilities.

        Query: {query}

        Instructions:
        - Use the history and context to maintain conversation continuity.
        - If the query references "my order" and an order ID exists in context, classify as mysql, reschedule_delivery, or address_change as appropriate.
        - General questions without specific order references are csv.
        - Respond with only the intent string (e.g., "csv", "general") and nothing else.
        - Do not include JSON, extra text, or explanations.

        Examples:
        - Query: "How do I track my package?" -> csv
        - Query: "Status of ORD123" -> mysql
        - Query: "Reschedule my delivery" -> reschedule_delivery
        - Query: "Change my address" -> address_change
        - Query: "Hi" -> general
        - Query: "What can you do?" -> capabilities
        Hello hi chit chat capbilities should be considered in relevant
        Irrelevant topics include general knowledge questions (e.g., "What is AI?") or unrelated subjects.

        Query: {query}

        Respond with "relevant" or "irrelevant".
        """
    )
    try:
        final_prompt = prompt_template.format(query=query)
        response = llm.invoke(final_prompt)
        print(f"RES:{response}")
        return response.content.strip().lower() == "relevant"
    except Exception as e:
        logger.error(f"Error checking query relevance: {e}")
        return False

def is_continuing_query(session_id: str, intent: str, query: str) -> bool:
    """Determine if the query continues an existing conversation."""
    if session_id not in session_context_cache:
        return False
    
    context = session_context_cache[session_id]
    last_intent = context.get("last_intent")
    
    if not last_intent:
        return False
    
    prompt_template = ChatPromptTemplate.from_template(
        """
        Determine if the current query continues the previous conversation based on the last intent and context.

        Last Intent: {last_intent}
        Current Intent: {current_intent}
        Current Query: {query}
        Recent Order IDs: {order_ids}
        Waiting for: {waiting_for}

        Instructions:
        - A query is continuing if it references the same intent (e.g., 'reschedule_delivery' after 'reschedule_delivery'),
          or if it provides a date or address in response to a prompt for 'reschedule_delivery' or 'address_change'.
        - For 'reschedule_delivery', a query continues if it mentions a date (e.g., 'tomorrow', 'May 20') or references the ongoing rescheduling request.
        - For 'address_change', a query continues if it provides an address or references the ongoing address change request.
        - For 'small_talks', a query continues if it responds to a previous small talk (e.g., 'Great' after 'How are you').
        - A query is new if it introduces a different intent or is unrelated to recent orders or prompts.
        - Return 'true' for continuing queries, 'false' for new queries.
        """
    )
    try:
        waiting_for = context.get("waiting_for", "")
        final_prompt = prompt_template.format(
            last_intent=last_intent,
            current_intent=intent,
            query=query,
            order_ids=", ".join(context["order_ids"]) if context["order_ids"] else "None",
            waiting_for=waiting_for or "None"
        )
        response = llm.invoke(final_prompt)
        print(f"RES :{response}")
        if response.content.strip().lower() == "false":
            with get_db_connection() as conn:
                sql_query = """
                    UPDATE chat_sessions SET last_order_id = NULL WHERE id = %s;
                """
                execute_query(conn, sql_query, (session_id,), fetch=False)
        return response.content.strip().lower() == "true"
    except Exception as e:
        logger.error(f"Error determining query type: {e}")
        return False
    
def handle_small_talks(session_id: str, query: str) -> Dict[str, str]:
    """Handle small talk queries like 'How are you', 'Great', 'Thanks', 'Good morning' using LLM."""
    try:
        prompt_template = ChatPromptTemplate.from_template(
            """
            You are a friendly assistant for a Transportation & Logistics company. The user has made a small talk query, such as greetings (excluding "Hi" or "Hello"), expressions of gratitude, or casual remarks.

            Query: {query}

            Instructions:
            - Respond in a friendly, conversational tone appropriate for small talk.
            - Keep the response concise, under 50 words.
            - Avoid logistics-related details unless the user mentions them.
            - Always include a gentle nudge to assist with logistics needs (e.g., "How can I help with your delivery?").
            - CRITICAL : Detect if the user's message is small talk (e.g., "thanks", "how are you", "bye") and reply appropriately.
                Respond to these common small talk types:
                - Greetings (hi, hello, hey, good morning)
                - Farewells (bye, goodbye, see you later)
                - Gratitude (thanks, thank you, appreciate it)
                - Politeness (how are you, how’s it going)
            - CRITICAL : NEVER respond to that are not related to Retail or Order.e.g.what is dog or what is ai.

    
            Respond with only the small talk response string.
            """
        )

        final_prompt = prompt_template.format(query=query)
        response = llm.invoke(final_prompt)
        response_text = response.content.strip()

        # Fallback response if LLM output is empty or invalid
        if not response_text:
            response_text = "Nice to chat! How can I assist with your logistics needs?"

        save_chat_message(session_id, 'user', query)
        save_chat_message(session_id, 'assistant', response_text)
        update_session_context(session_id, "small_talks", query)
        return {"response": response_text}
    except Exception as e:
        logger.error(f"Error handling small talk query: {e}")
        response_text = "Nice to chat! How can I assist with your logistics needs?"
        save_chat_message(session_id, 'user', query)
        save_chat_message(session_id, 'assistant', response_text)
        update_session_context(session_id, "small_talks", query)
        return {"response": response_text}
    
df = pd.read_csv('faqs.csv')  # Assume columns: 'question', 'answer'
questions = df['question'].tolist()

# Create embeddings
embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_texts(questions, embeddings)

def intent_classifier(query: str, session_id: str) -> str:
    """Classify the intent of the query, returning only the intent string."""
    if len(session_context_cache) > 100:
        clean_old_contexts()

    results = vector_store.similarity_search_with_score(query, k=1)
    if results[0][1] > 0.8:  # Adjust threshold as needed
        return "csv"
    
    try:
        context = retrieve_chat_history(session_id)
        chat_history = context["messages"]
    except Exception as e:
        logger.warning(f"Failed to retrieve chat history: {e}")
        chat_history = []
        context = {
            "order_ids": set(),
            "last_order_id": None,
            "email": None,
            "last_intent": None,
            "waiting_for": None
        }
    
    prompt_template = ChatPromptTemplate.from_template(
        """
        You are an intent classifier for a Transportation & Logistics assistant. Classify the query into one of the following intents:

        - csv: General FAQs about placing order, payments, shipping, delivery, warranties, returns, dealers, or technical support.
        - mysql: Queries about specific customer data (e.g., order details, order status, delivery dates, invoices) or actions requiring an order ID or email. Only classify as mysql when the user asks for shipment or invoice details.
        - reschedule_delivery: Requests to change delivery dates or times, or follow-up queries providing a date after being prompted.
        - address_change: Requests to update delivery addresses, or follow-up queries providing an address after being prompted.
        - general: Greetings specifically limited to "Hi" or "Hello".
        - small_talks: Small talk queries like "How are you", "Great", "Thanks", "Good morning", or similar casual phrases, excluding "Hi" and "Hello".
        - capabilities: Questions about the assistant's capabilities (e.g., "What can you do?").

        Query: {query}
        Recent Order IDs: {order_ids}
        Last Intent: {last_intent}
        Waiting for: {waiting_for}

        Instructions:
        - Use the history and context to maintain conversation continuity.
        - Placing the order only goes in "csv"; other order-related queries go in "mysql".
        - If the query references "my order" and an order ID exists in context, classify as mysql, reschedule_delivery, or address_change as appropriate.
        - If waiting_for is 'date' and the query contains a date (e.g., 'tomorrow', '2025-05-20'), classify as reschedule_delivery.
        - If waiting_for is 'address' and the query contains an address, classify as address_change.
        - General questions without specific order references are csv.
        - Classify queries like "How are you", "Great", "Thanks", "Good morning" as small_talks, but "Hi" or "Hello" as general.
        - Respond with only the intent string (e.g., "csv", "small_talks") and nothing else.
        - Do not include JSON, extra text, or explanations.
        - Valid intents: csv, mysql, reschedule_delivery, address_change, general, small_talks, capabilities.

        Examples:
        - Query: "How do I track my package?" -> csv
        - Query: "Status of ORD123" -> mysql
        - Query: "Reschedule my delivery" -> reschedule_delivery
        - Query: "tomorrow" (waiting_for=date) -> reschedule_delivery
        - Query: "Change my address" -> address_change
        - Query: "123 Main St, Springfield, IL" (waiting_for=address) -> address_change
        - Query: "Hi" -> general
        - Query: "Hello" -> general
        - Query: "How are you" -> small_talks
        - Query: "Good morning" -> small_talks
        - Query: "Thanks" -> small_talks
        - Query: "What can you do?" -> capabilities
        """
    )
    
    valid_intents = {"csv", "mysql", "reschedule_delivery", "address_change", "general", "small_talks", "capabilities"}
    
    try:
        kwargs = {
            "query": query,
            "order_ids": ", ".join(context["order_ids"]) if context["order_ids"] else "None",
            "last_intent": context["last_intent"] or "None",
            "waiting_for": context.get("waiting_for", "None")
        }
        logger.debug(f"Prompt kwargs: {kwargs}")
        
        final_prompt = prompt_template.format(**kwargs)
        logger.debug(f"Formatted prompt: {final_prompt}")
        
        response = llm.invoke(final_prompt)
        intent = response.content.strip()
        logger.debug(f"Raw LLM response for intent classification: {intent}")
        
        if intent not in valid_intents:
            logger.warning(f"Invalid intent returned: {intent}")
            return "general"
        
        update_session_context(session_id, intent, query)
        logger.info(f"Classified intent: {intent} for query: {query}")
        return intent
    except Exception as e:
        logger.error(f"Error in intent classification: {str(e)}")
        return "general"


def extract_delivery_date(session_id: str, query: str) -> str:
    """Extract delivery date from current query or recent chat history using LLM."""
    try:
        current_time = datetime.now()
        current_year = current_time.year
    
        print(f"QUERYY:{query}")
        prompt_template = ChatPromptTemplate.from_template(
            """
            You are an assistant for a Transport & Logistics company. Your task is to extract a delivery date from the provided query.

Query: {query}

Instructions:
- Examine the provided query for an explicit delivery date in recognizable formats, such as:
  - 'today' (return: {today} in YYYY-MM-DD format)
  - 'tomorrow' (return: {tomorrow} in YYYY-MM-DD format)
  - Specific dates like 'May 20', '2025-05-20', '20th May', '20 May 2025', etc.
- If a specific date is found without a year, assume the year is {current_year}.
- CRITICAL: Only extract a date explicitly stated in the query. Do not use dates from logs, context, or external metadata.
- CRITICAL: If no recognizable delivery date is found in the query, return exactly: "" (empty string).
- CRITICAL: For 'yesterday' or any past date, return: "" (empty string).
- CRITICAL: Do not assume dates like 'today' or 'tomorrow' unless explicitly mentioned in the query.
- Response must be in the format 'YYYY-MM-DD' for valid dates or "" for no date. No labels, prefixes, or explanations.

Response format: 
- Valid date example: 2025-05-20
- No date or invalid/past date: ""
            """
        )
        
        today_str = current_time.strftime('%Y-%m-%d')
        tomorrow_str = (current_time + timedelta(days=1)).strftime('%Y-%m-%d')
        
        final_prompt = prompt_template.format(
            query=query,
            current_year=current_year,
            today=today_str,
            tomorrow=tomorrow_str
        )
        response = llm.invoke(final_prompt)
        date_str = response.content.strip()
        
        # Basic validation: ensure the date matches YYYY-MM-DD format
        # try:
        #     if date_str:
        #         parsed_date = datetime.strptime(date_str, '%Y-%m-%d')
        #         # Ensure the date is today or in the future
        #         if parsed_date.date() < current_time.date():
        #             date_str = ""
        # except ValueError:
        #     date_str = ""

        print(f"Date: {date_str}")
        
        return date_str
    except Exception as e:
        logger.error(f"Error extracting delivery date: {e}")
        return ""

def handle_reschedule_delivery(session_id: str, query: str, chat_history: Optional[List] = None) -> Dict[str, Any]:
    """Handle delivery rescheduling requests."""
    if chat_history is None:
        chat_history = [AIMessage(content="Hello! I can help with rescheduling your delivery.")]
    
    chat_history, order_id = format_chat_history_and_extract_order_id(session_id, query)
    
    if not order_id:
        response = "Could you please share your valid order ID, so I can check the details for you?"
        save_chat_message(session_id, 'user', query)
        save_chat_message(session_id, 'assistant', response)
        update_session_context(session_id, "reschedule_delivery", query, waiting_for="order_id")
        return {"response": response}
    
    try:
        conn = get_db_connection(MYSQL_QUERY_CONFIG)
        if not conn:
            logger.error("Database connection failed for reschedule eligibility check")
            return {"error": "Database connection failed", "error_code": "DB_CONNECTION_FAILED"}
        
        query_check = "SELECT reschedule_eligible, expected_delivery, shipment_status FROM orders WHERE order_id = %s"
        result = execute_query(conn, query_check, (order_id,), fetch=True)
        
        if not result:
            response = f"Order {order_id} not found. Please verify the order ID and try again."
            save_chat_message(session_id, 'user', query)
            save_chat_message(session_id, 'assistant', response)
            return {"response": response}
        
        order_details = result[0]
        
        if not order_details['reschedule_eligible']:
            response = f"Order {order_id} can no longer be rescheduled.\n If you need further assistance, please contact our support team."
            save_chat_message(session_id, 'user', query)
            save_chat_message(session_id, 'assistant', response)
            update_session_context(session_id, "reschedule_delivery", query, order_id)
            return {"response": response}
        
        date_str = extract_delivery_date(session_id, query)
        
        if not date_str or date_str == '""':
            current_date = order_details['expected_delivery']
            response = f"Please provide the new delivery date for order {order_id} (current date: {current_date})."
            save_chat_message(session_id, 'user', query)
            save_chat_message(session_id, 'assistant', response)
            update_session_context(session_id, "reschedule_delivery", query, order_id, waiting_for="date")
            return {"response": response}
        
        try:
            new_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            if new_date <= datetime.now().date():
                response = "I’m sorry, but rescheduling is only possible for future dates. Could you please provide a valid future date?"
                save_chat_message(session_id, 'user', query)
                save_chat_message(session_id, 'assistant', response)
                update_session_context(session_id, "reschedule_delivery", query, order_id, waiting_for="date")
                return {"response": response}
            
            if (new_date - datetime.now().date()).days > 30:
                response = "To ensure timely processing, rescheduling is limited to dates within the next 30 days. Please choose a date within that range."
                save_chat_message(session_id, 'user', query)
                save_chat_message(session_id, 'assistant', response)
                update_session_context(session_id, "reschedule_delivery", query, order_id, waiting_for="date")
                return {"response": response}
            
            update_query = "UPDATE orders SET expected_delivery = %s WHERE order_id = %s"
            execute_query(conn, update_query, (new_date, order_id), fetch=False)
            
            response = f"The delivery for Order {order_id} has been rescheduled to {new_date}. \n Is there anything else I can help you with?"
            save_chat_message(session_id, 'user', query)
            save_chat_message(session_id, 'assistant', response)
            update_session_context(session_id, "reschedule_delivery", query, order_id, waiting_for=None)
            return {"response": response}
        except ValueError:
            response = f"Please provide the new delivery date for order {order_id} in a valid format (e.g., '2025-05-20' or 'tomorrow')."
            save_chat_message(session_id, 'user', query)
            save_chat_message(session_id, 'assistant', response)
            update_session_context(session_id, "reschedule_delivery", query, order_id, waiting_for="date")
            return {"response": response}
    except Exception as e:
        logger.error(f"Error in reschedule delivery: {e}")
        response = "An error occurred while processing your request. Please try again or contact support."
        save_chat_message(session_id, 'user', query)
        save_chat_message(session_id, 'assistant', response)
        update_session_context(session_id, "reschedule_delivery", query, order_id)
        return {"response": response}
    finally:
        if conn and conn.is_connected():
            conn.close()


def extract_delivery_address(session_id: str, query: str) -> str:
    """Extract delivery address from current query or recent chat history using LLM."""
    try:
        
        
        prompt_template = ChatPromptTemplate.from_template(
    """
    You are an assistant for a Transport & Logistics company. Your task is to extract a delivery address from the current user query or recent conversation history.

    A delivery address typically includes:
    - Street name or number
    - City
    - State
    - Postal code (ZIP or PIN)

    Instructions:
    - Look in the current query first.
    - Only return the address string.
    - If no address is found, return an empty string.
    - Do not explain or add any extra text.

    Current Query: {query}

    Respond with just the address, or "".
    """
)

        
        final_prompt = prompt_template.format(query=query)
        response = llm.invoke(final_prompt)
        print(f"RES: {response}")
        address = response.content.strip()

        print(f"Add_1: {address}")
        
        # Basic validation: address should be reasonably long and contain digits (for ZIP code)
        if address and (len(address) < 10 or not any(char.isdigit() for char in address)):
            print("Here")
            address = ""

        print(f"Add : {address}")
        
        return address
    except Exception as e:
        logger.error(f"Error extracting delivery address: {e}")
        return ""

def handle_address_change(session_id: str, query: str, chat_history: Optional[List] = None) -> Dict[str, Any]:
    """Handle address change requests."""
    if chat_history is None:
        chat_history = [AIMessage(content="Hello! I can help with changing your delivery address.")]
    
    chat_history, order_id = format_chat_history_and_extract_order_id(session_id, query)
    
    if not order_id:
        response = "Could you please share your valid order ID, so I can check the details for you?"
        save_chat_message(session_id, 'user', query)
        save_chat_message(session_id, 'assistant', response)
        update_session_context(session_id, "address_change", query, waiting_for="order_id")
        return {"response": response}
    
    try:
        conn = get_db_connection(MYSQL_QUERY_CONFIG)
        if not conn:
            logger.error("Database connection failed for address change eligibility check")
            return {"error": "Database connection failed", "error_code": "DB_CONNECTION_FAILED"}
        
        query_check = "SELECT address_change_eligible, delivery_address, shipment_status FROM orders WHERE order_id = %s"
        result = execute_query(conn, query_check, (order_id,), fetch=True)
        
        if not result:
            response = f"Order {order_id} not found. Please verify the order ID and try again."
            save_chat_message(session_id, 'user', query)
            save_chat_message(session_id, 'assistant', response)
            return {"response": response}
        
        order_details = result[0]
        
        if not order_details['address_change_eligible']:
            response = f"Order {order_id} isn’t eligible for an address change at this stage.\n If you need further assistance, please contact our support team."
            save_chat_message(session_id, 'user', query)
            save_chat_message(session_id, 'assistant', response)
            update_session_context(session_id, "address_change", query, order_id)
            return {"response": response}
        
        new_address = extract_delivery_address(session_id, query)
        
        if not new_address:
            current_address = order_details['delivery_address']
            response = f"Please share the new delivery address for Order {order_id}. The current address on record is: {current_address}."
            save_chat_message(session_id, 'user', query)
            save_chat_message(session_id, 'assistant', response)
            update_session_context(session_id, "address_change", query, order_id, waiting_for="address")
            return {"response": response}
        
        update_query = "UPDATE orders SET delivery_address = %s WHERE order_id = %s"
        execute_query(conn, update_query, (new_address, order_id), fetch=False)
        
        response = f"The address for {order_id} has been updated to:\n  {new_address}. \n Is there anything else I can help you with?"
        save_chat_message(session_id, 'user', query)
        save_chat_message(session_id, 'assistant', response)
        update_session_context(session_id, "address_change", query, order_id, waiting_for=None)
        return {"response": response}
    except Exception as e:
        logger.error(f"Error in address change: {e}")
        response = "An error occurred while processing your request. Please try again or contact support."
        save_chat_message(session_id, 'user', query)
        save_chat_message(session_id, 'assistant', response)
        update_session_context(session_id, "address_change", query, order_id)
        return {"response": response}
    finally:
        if conn and conn.is_connected():
            conn.close()

def handle_general_query(session_id: str, query: str) -> Dict[str, str]:
    """Handle general greetings or unrelated queries."""
    normalized_query = query.lower().strip()
    greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "how are you", "who are you"]
    
    if any(greeting in normalized_query for greeting in greetings):
        response = """
Hi! I’m your delivery assistant.\n
I can help you with the following:\n
- Track your shipment\n
- Reschedule a delivery\n
- Update your address\n
- Get your invoice\n
- Answer common questions\n
\n
Just tell me what you’d like help with!

            """
    else:
        response = """
Hi! I’m your delivery assistant.\n
I can help you with the following:\n
- Track your shipment\n
- Reschedule a delivery\n
- Update your address\n
- Get your invoice\n
- Answer common questions\n
\n
Just tell me what you’d like help with!
"""
    
    save_chat_message(session_id, 'user', query)
    save_chat_message(session_id, 'assistant', response)
    update_session_context(session_id, "general", query)
    return {"response": response}

def handle_capabilities_query(session_id: str, query: str) -> Dict[str, str]:
    """Handle queries about assistant capabilities."""
    response = (
        """
        I can assist you with the following:\n
        - Track your shipment\n
        - Reschedule a delivery\n
        - Update your address\n
        - Get your invoice\n
        - Answer common questions\n
        \n
        Just tell me what you’d like help with!

            """
    )
    
    save_chat_message(session_id, 'user', query)
    save_chat_message(session_id, 'assistant', response)
    update_session_context(session_id, "capabilities", query)
    return {"response": response}

def chat_with_csv(session_id: str, query: str) -> Dict[str, Any]:
    """Handle CSV-based FAQ queries."""
    global vector_store
    try:
        if vector_store is None:
            if os.path.exists(FAISS_PATH):
                embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=os.getenv("OPENAI_API_KEY"))

                vector_store = FAISS.load_local(
                    FAISS_PATH,
                    embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info("Loaded existing FAISS index")
            else:
                logger.warning("No CSV data available")
                return {"error": "No CSV data uploaded", "error_code": "NO_DATA"}
        
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=os.getenv("OPENAI_API_KEY"))
        faiss_index = FAISS.load_local(FAISS_PATH, embeddings,allow_dangerous_deserialization=True)
        def retrieve_documents(query, k=5):
            """Retrieves top-k most relevant documents from FAISS."""
            docs = faiss_index.similarity_search(query, k=k)
            return "\n".join([doc.page_content for doc in docs]) if docs else "No relevant data found."

        
        context = retrieve_documents(query)
        
        prompt_template = ChatPromptTemplate.from_template(
            """
            You are a logistics assistant answering FAQs based on provided data.
            Context: {context}
            Query: {query}
            Provide a concise answer based only on the context. If the answer is not in the context, say so politely.
            Never reveal raw data or mention the source.
            """
        )
        
        final_prompt = prompt_template.format(context=context, query=query)
        response = llm.invoke(final_prompt)
        response_text = response.content.strip()
        
        if not response_text:
            response_text = "I don't have enough information to answer that. Please provide more details or ask about something else."
        
        save_chat_message(session_id, 'user', query)
        save_chat_message(session_id, 'assistant', response_text)
        update_session_context(session_id, "csv", query)
        return {"response": response_text}
    except Exception as e:
        logger.error(f"CSV query error: {e}")
        response = "An error occurred while processing your FAQ query. Please try again."
        save_chat_message(session_id, 'user', query)
        save_chat_message(session_id, 'assistant', response)
        return {"response": response}

def chat_with_mysql(session_id: str, query: str, chat_history: Optional[List] = None) -> Dict[str, Any]:
    """Handle MySQL database queries."""
    if chat_history is None:
        chat_history = [AIMessage(content="Hello! I can help with order-specific queries.")]
    print(f"QUE : {query}")
    formatted_history, order_id = format_chat_history_and_extract_order_id(session_id, query)

    if not order_id:
        response = "Could you please share your valid order ID, so I can check the details for you?"
        save_chat_message(session_id, 'user', query)
        save_chat_message(session_id, 'assistant', response)
        update_session_context(session_id, "mysql", query, waiting_for="order_id")
        return {"response": response}
    
    try:
        db_uri = f"mysql+mysqlconnector://{MYSQL_QUERY_CONFIG['user']}:{MYSQL_QUERY_CONFIG['password']}@{MYSQL_QUERY_CONFIG['host']}:{MYSQL_QUERY_CONFIG['port']}/{MYSQL_QUERY_CONFIG['database']}"
        db = sql_database.SQLDatabase.from_uri(db_uri)
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return {"error": f"Database connection failed: {str(e)}", "error_code": "DB_CONNECTION_FAILED"}
    
    def get_sql(schema: str, formatted_history: str, query: str, order_id: Optional[str]) -> str:
        """Generate SQL query."""
        context = session_context_cache.get(session_id, {})
        email = context.get("email")
        
        if not order_id and not email:
            return "Could you please share your valid order ID, so I can check the details for you?"
        
        context_info = f"Order ID: {order_id}\n" if order_id else f"Email: {email}\n"
        prompt_template = ChatPromptTemplate.from_template(
            """
            You are a logistics assistant. The 'orders' table has columns: order_id, customer_name, email, status, expected_delivery, delivery_address, is_active, has_invoice, invoice_url, created_at, reschedule_eligible, address_change_eligible.
            You need to classify that wheater the user asks for the shipment or invoice.
            If the query is related to the invoice only than classify in 'invoice'.
            If details of the order is asked just return shipment.
            CRITICAL : If you are unable to find that than simply classify in the shipment.
            Response must be shipment or invoice nothing else.
            """
        )
        try:
            final_prompt = prompt_template.format(
                query=query,
                schema=schema,
                history=formatted_history,
                context_info=context_info
            )
            response = llm.invoke(final_prompt)
            print(f"SQL QUERY : {response.content.strip()}")
            response_sql = response.content.strip()
            if "invoice" in response_sql:
                return f"SELECT invoice_url FROM orders WHERE order_id = '{order_id}';"
            elif "shipment":
                return f"SELECT customer_name, email,shipment_status,expected_delivery , delivery_address FROM orders WHERE order_id = '{order_id}';"
            else :
                return f"SELECT * FROM orders WHERE order_id = '{order_id}';"
            
        except Exception as e:
            logger.error(f"SQL generation error: {e}")
            return f"Error generating SQL query: {str(e)}"
    
    def get_response(schema: str, formatted_history: str, query: str, sql_query: str, sql_response: str) -> str:
        """Generate natural language response."""
        prompt_template = ChatPromptTemplate.from_template(
            """
            You are a logistics assistant. provide a natural language response.
            Query: {query}
            Schema: {schema}
            History: {history}
            SQL Query: {sql_query}
            SQL Response: {sql_response}
            Use 'according to my knowledge' if appropriate. If no data, suggest providing more details.
            """
        )
        try:
            final_prompt = prompt_template.format(
                query=query,
                schema=schema,
                history=formatted_history,
                sql_query=sql_query,
                sql_response=sql_response
            )
            response = llm.invoke(final_prompt)
            return response.content.strip()
        except Exception as e:
            logger.error(f"Response generation error: {e}")
            return "An error occurred while generating the response."
    
    try:
        chat_history.append(HumanMessage(content=query))
        sql_query = get_sql(db.get_table_info(), formatted_history, query, order_id)
        
        if sql_query.startswith("Please provide"):
            save_chat_message(session_id, 'user', query)
            save_chat_message(session_id, 'assistant', sql_query)
            update_session_context(session_id, "mysql", query, order_id)
            return {"response": sql_query}
        
        try:
            sql_response = db.run(sql_query)
        except Exception as e:
            logger.error(f"SQL execution error: {e}")
            response = "Sorry, I encountered an error. Please try again or refine your question."
            save_chat_message(session_id, 'user', query)
            save_chat_message(session_id, 'assistant', response)
            return {"response": response, "sql_query": sql_query, "sql_response": str(e)}
        
        natural_language_response = get_response(db.get_table_info(), formatted_history, query, sql_query, sql_response)
        
        save_chat_message(session_id, 'user', query)
        save_chat_message(session_id, 'assistant', natural_language_response)
        update_session_context(session_id, "mysql", query, order_id)
        
        return {
            "response": natural_language_response,
            "sql_query": sql_query,
            "sql_response": sql_response
        }
    except Exception as e:
        logger.error(f"Unexpected error in MySQL query: {e}")
        response = "An unexpected error occurred. Please try again or contact support."
        save_chat_message(session_id, 'user', query)
        save_chat_message(session_id, 'assistant', response)
        return {"response": response}

@app.route('/start_session', methods=['POST'])
def start_session():
    """Start a new session."""
    try:
        session_id = create_session()
        return jsonify({"status": "success", "session_id": session_id}), 201
    except Exception as e:
        logger.error(f"Session creation error: {e}")
        return jsonify({"error": str(e), "error_code": "SESSION_CREATION_FAILED"}), 500

@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    """Upload and process CSV file."""
    session_id = request.form.get('session_id')
    if not session_id:
        logger.warning("Missing session ID in CSV upload")
        return jsonify({"error": "Session ID is required", "error_code": "MISSING_SESSION_ID"}), 400
    
    if 'file' not in request.files:
        logger.warning("No file part in CSV upload request")
        return jsonify({"error": "No file part in the request", "error_code": "NO_FILE"}), 400
    
    file = request.files['file']
    if file.filename == '':
        logger.warning("No file selected in CSV upload")
        return jsonify({"error": "No file selected", "error_code": "NO_FILE_SELECTED"}), 400
    
    if file and file.filename.lower().endswith('.csv'):
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        try:
            file.save(file_path)
            process_csv(file_path)
            logger.info(f"Successfully processed CSV: {filename}")
            return jsonify({"message": "CSV processed successfully"}), 200
        except Exception as e:
            logger.error(f"CSV processing error: {e}")
            return jsonify({"error": str(e), "error_code": "PROCESSING_FAILED"}), 500
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)
    else:
        logger.warning("Invalid file format in CSV upload")
        return jsonify({"error": "Please upload a CSV file", "error_code": "INVALID_FILE"}), 400

@app.route('/query', methods=['POST'])
def query_data():
    """Handle user queries."""
    try:
        data = request.get_json()
        if not data or 'session_id' not in data or 'query' not in data:
            logger.warning("Missing parameters in query request")
            return jsonify({"error": "Session ID and query are required", "error_code": "MISSING_PARAMETERS"}), 400

        session_id = data['session_id']
        user_input = data['query'].strip()
        if not user_input:
            logger.warning("Empty query received")
            return jsonify({"error": "Query cannot be empty", "error_code": "EMPTY_QUERY"}), 400
        
        logger.info(f"Processing query: '{user_input}' for session: {session_id}")
        
        intent = intent_classifier(user_input, session_id)
        logger.debug(f"Classified intent: {intent} for query: {user_input}")
        
        # Check if the query is continuing an existing conversation
        if intent not in ["general", "capabilities", "small_talks"] and is_continuing_query(session_id, intent, user_input):
            print(f"Continuing query detected for session {session_id}")
        else:
            if session_context_cache.get(session_id, {}).get("last_intent") and intent != session_context_cache[session_id]["last_intent"]:
                response = f"Would you like to switch to a new topic ({intent}) or continue with the previous request?"
                save_chat_message(session_id, 'user', user_input)
                save_chat_message(session_id, 'assistant', response)
                update_session_context(session_id, intent, user_input)
                return jsonify({"response": response}), 200
        
        chat_history = retrieve_chat_history(session_id)["messages"]
        
        if intent == "csv":
            result = chat_with_csv(session_id, user_input)
        elif intent == "mysql":
            result = chat_with_mysql(session_id, user_input, chat_history)
        elif intent == "reschedule_delivery":
            result = handle_reschedule_delivery(session_id, user_input, chat_history)
        elif intent == "address_change":
            result = handle_address_change(session_id, user_input, chat_history)
        elif intent == "general":
            result = handle_general_query(session_id, user_input)
        elif intent == "capabilities":
            result = handle_capabilities_query(session_id, user_input)
        elif intent == "small_talks":
            result = handle_small_talks(session_id, user_input)
        else:
            logger.warning(f"Unknown intent: {intent}")
            result = {"response": "I'm not sure how to handle that request. Please ask about orders or logistics."}
        
        if "error" in result:
            logger.error(f"Query processing error: {result['error']}")
            return jsonify({"error": result["error"], "error_code": result["error_code"]}), 500
        
        response = {"response": result["response"]}
        if "sql_query" in result:
            response.update({
                "sql_query": result["sql_query"],
                "sql_response": result["sql_response"]
            })
        
        logger.info(f"Query processed successfully for session {session_id}")
        return jsonify(response), 200
    except Exception as e:
        logger.error(f"Unexpected error in query processing: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e), "error_code": "UNEXPECTED_ERROR"}), 500

@app.route('/chat_history/<session_id>', methods=['GET'])
def get_chat_history_endpoint(session_id: str):
    """Retrieve chat history for a session."""
    try:
        history_data = retrieve_chat_history(session_id)
        formatted_messages = [
            {
                "role": "user" if isinstance(msg, HumanMessage) else "assistant",
                "content": msg.content,
                "timestamp": msg.created_at.isoformat() if hasattr(msg, 'created_at') else datetime.now().isoformat()
            }
            for msg in history_data["messages"]
        ]
        return jsonify({"messages": formatted_messages}), 200
    except Exception as e:
        logger.error(f"Chat history retrieval error: {e}")
        if str(e) == "Session not found or deleted":
            return jsonify({"error": "Session not found or deleted", "error_code": "INVALID_SESSION"}), 404
        return jsonify({"error": str(e), "error_code": "HISTORY_RETRIEVAL_FAILED"}), 500

@app.route('/clear_session', methods=['POST'])
def clear_session():
    """Clear chat history for a session."""
    try:
        data = request.get_json()
        if not data or 'session_id' not in data:
            logger.warning("Missing session ID in clear session request")
            return jsonify({"error": "Session ID is required", "error_code": "MISSING_SESSION_ID"}), 400
        
        session_id = data['session_id']
        if not mark_session_as_deleted(session_id):
            logger.error(f"Failed to clear session: {session_id}")
            return jsonify({"error": "Failed to clear session", "error_code": "SESSION_CLEAR_FAILED"}), 500
        
        logger.info(f"Session cleared: {session_id}")
        return jsonify({"message": "Session cleared successfully"}), 200
    except Exception as e:
        logger.error(f"Clear session error: {e}")
        return jsonify({"error": str(e), "error_code": "SESSION_CLEAR_FAILED"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    try:
        db_status = False
        mysql_db_status = False
        
        conn = get_db_connection()
        if conn and conn.is_connected():
            db_status = True
            conn.close()
        
        mysql_conn = get_db_connection(config=MYSQL_QUERY_CONFIG)
        if mysql_conn and mysql_conn.is_connected():
            mysql_db_status = True
            mysql_conn.close()
        
        logger.info("Health check performed")
        return jsonify({
            "status": "healthy",
            "session_database": "connected" if db_status else "disconnected",
            "mysql_query_database": "connected" if mysql_db_status else "disconnected"
        }), 200
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return jsonify({"error": str(e), "error_code": "HEALTH_CHECK_FAILED"}), 500
    
@app.route('/')
def home():
    return {"message": "TNL Flask API running!"}

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)