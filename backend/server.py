from fastapi import FastAPI, APIRouter, HTTPException, Depends, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import os
import logging
import asyncio
import aiomysql
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Float, Date, DateTime, Boolean, Text
from datetime import datetime, date
import uuid
import json
import pandas as pd
import io

# Load environment variables
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# Database configuration
MYSQL_HOST = os.getenv('MYSQL_HOST', 'localhost')
MYSQL_PORT = int(os.getenv('MYSQL_PORT', 3306))
MYSQL_USER = os.getenv('MYSQL_USER', 'ecommerce')
MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD', '')
MYSQL_DATABASE = os.getenv('MYSQL_DATABASE', 'ecommerce_ai')

# Create MySQL connection string
DATABASE_URL = f"mysql+aiomysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}"

# SQLAlchemy setup
Base = declarative_base()

# Database Models
class AdSales(Base):
    __tablename__ = "ad_sales"
    
    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, nullable=False)
    product_id = Column(String(100), nullable=False)
    product_name = Column(String(255), nullable=False)
    ad_spend = Column(Float, default=0.0)
    clicks = Column(Integer, default=0)
    impressions = Column(Integer, default=0)
    ad_sales = Column(Float, default=0.0)
    cpc = Column(Float, default=0.0)  # Cost Per Click
    cpm = Column(Float, default=0.0)  # Cost Per Mille
    ctr = Column(Float, default=0.0)  # Click Through Rate
    acos = Column(Float, default=0.0)  # Advertising Cost of Sales
    roas = Column(Float, default=0.0)  # Return on Ad Spend

class TotalSales(Base):
    __tablename__ = "total_sales"
    
    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, nullable=False)
    product_id = Column(String(100), nullable=False)
    product_name = Column(String(255), nullable=False)
    total_sales = Column(Float, default=0.0)
    units_sold = Column(Integer, default=0)
    price_per_unit = Column(Float, default=0.0)
    total_revenue = Column(Float, default=0.0)
    organic_sales = Column(Float, default=0.0)
    sessions = Column(Integer, default=0)
    conversion_rate = Column(Float, default=0.0)

class ProductEligibility(Base):
    __tablename__ = "product_eligibility"
    
    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(String(100), nullable=False, unique=True)
    product_name = Column(String(255), nullable=False)
    category = Column(String(100))
    brand = Column(String(100))
    is_eligible_for_ads = Column(Boolean, default=True)
    is_active = Column(Boolean, default=True)
    created_date = Column(DateTime, default=datetime.utcnow)
    updated_date = Column(DateTime, default=datetime.utcnow)

# Chat History Model
class ChatHistory(Base):
    __tablename__ = "chat_history"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(100), nullable=False, index=True)
    user_question = Column(Text, nullable=False)
    sql_query = Column(Text, nullable=False)
    ai_response = Column(Text, nullable=False)
    query_results = Column(Text)  # JSON string of results
    timestamp = Column(DateTime, default=datetime.utcnow)
    user_id = Column(String(100), default="default_user")  # For multi-user support later

# Data Upload Log Model  
class DataUploadLog(Base):
    __tablename__ = "data_upload_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    table_name = Column(String(100), nullable=False)
    records_count = Column(Integer, default=0)
    upload_timestamp = Column(DateTime, default=datetime.utcnow)
    status = Column(String(50), default="success")
    error_message = Column(Text)

# Pydantic Models
class QueryRequest(BaseModel):
    question: str
    session_id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = "default_user"

class QueryResponse(BaseModel):
    question: str
    sql_query: str
    results: List[Dict[str, Any]]
    human_readable_answer: str
    session_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class ChatMessage(BaseModel):
    id: int
    session_id: str
    user_question: str
    sql_query: str
    ai_response: str
    timestamp: datetime

class DataUploadRequest(BaseModel):
    table_name: str  # 'ad_sales', 'total_sales', or 'product_eligibility'
    data: List[Dict[str, Any]]

class UploadResponse(BaseModel):
    message: str
    records_uploaded: int
    filename: str
    table_name: str

# Create FastAPI app
app = FastAPI(title="E-commerce Data Query AI Agent", version="2.0.0")
api_router = APIRouter(prefix="/api")

# Database connection pool
async def get_db_connection():
    connection = await aiomysql.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        db=MYSQL_DATABASE,
        autocommit=True
    )
    return connection

# AI Query Processing Class
class AIQueryProcessor:
    def __init__(self):
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        self.gemini_model = os.getenv('GEMINI_MODEL', 'gemini-2.0-flash')
        
    async def setup_gemini(self):
        """Setup Gemini AI connection"""
        try:
            from emergentintegrations.llm.chat import LlmChat, UserMessage
            
            self.chat = LlmChat(
                api_key=self.gemini_api_key,
                session_id="ecommerce-ai-agent",
                system_message="""You are an expert SQL query generator for e-commerce data analysis. 
                You have access to three main tables:

                1. ad_sales table:
                   - date, product_id, product_name, ad_spend, clicks, impressions, ad_sales, cpc, cpm, ctr, acos, roas

                2. total_sales table:
                   - date, product_id, product_name, total_sales, units_sold, price_per_unit, total_revenue, organic_sales, sessions, conversion_rate

                3. product_eligibility table:
                   - product_id, product_name, category, brand, is_eligible_for_ads, is_active, created_date, updated_date

                Generate ONLY valid MySQL SQL queries. Return only the SQL query, no explanations or formatting.
                Use proper JOIN statements when data from multiple tables is needed.
                Always use appropriate aggregations (SUM, AVG, COUNT) for metrics.
                Use proper date filtering when time periods are mentioned.
                """
            ).with_model("gemini", self.gemini_model).with_max_tokens(2048)
            
            return True
        except Exception as e:
            logging.error(f"Failed to setup Gemini: {str(e)}")
            return False
    
    async def generate_sql(self, question: str) -> str:
        """Generate SQL query from natural language"""
        try:
            from emergentintegrations.llm.chat import UserMessage
            
            prompt = f"""
            Convert this business question to a SQL query: "{question}"
            
            Consider these common patterns:
            - "total sales" -> SUM(total_sales) or SUM(total_revenue)
            - "return on ad spend" or "ROAS" -> SUM(ad_sales)/SUM(ad_spend) or AVG(roas)
            - "cost per click" or "CPC" -> SUM(ad_spend)/SUM(clicks) or AVG(cpc)
            - "best performing" -> ORDER BY metric DESC LIMIT 10
            - "worst performing" -> ORDER BY metric ASC LIMIT 10
            
            Return only the SQL query.
            """
            
            user_message = UserMessage(text=prompt)
            response = await self.chat.send_message(user_message)
            
            # Clean up the response to extract just the SQL
            sql_query = response.strip()
            if sql_query.startswith('```sql'):
                sql_query = sql_query[6:]
            if sql_query.endswith('```'):
                sql_query = sql_query[:-3]
            
            return sql_query.strip()
            
        except Exception as e:
            logging.error(f"Error generating SQL: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to generate SQL: {str(e)}")
    
    async def execute_query(self, sql_query: str) -> List[Dict[str, Any]]:
        """Execute SQL query and return results"""
        connection = None
        try:
            connection = await get_db_connection()
            async with connection.cursor(aiomysql.DictCursor) as cursor:
                await cursor.execute(sql_query)
                results = await cursor.fetchall()
                return [dict(row) for row in results]
        except Exception as e:
            logging.error(f"Error executing query: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Query execution failed: {str(e)}")
        finally:
            if connection:
                connection.close()
    
    async def format_human_readable(self, question: str, results: List[Dict[str, Any]]) -> str:
        """Format results into human-readable answer"""
        try:
            from emergentintegrations.llm.chat import UserMessage
            
            if not results:
                return "No data found for your query."
            
            # Convert results to a readable format
            results_text = json.dumps(results[:10], indent=2, default=str)  # Limit to first 10 rows
            
            prompt = f"""
            Based on this business question: "{question}"
            And these query results: {results_text}
            
            Provide a clear, concise business-friendly answer. Include key metrics and insights.
            If there are multiple rows, summarize the key findings.
            Use business language, not technical database terms.
            """
            
            user_message = UserMessage(text=prompt)
            response = await self.chat.send_message(user_message)
            return response.strip()
            
        except Exception as e:
            logging.error(f"Error formatting response: {str(e)}")
            # Fallback to simple formatting
            if results:
                return f"Found {len(results)} results. Sample: {str(results[0])}"
            return "No results found."

# Initialize AI processor
ai_processor = AIQueryProcessor()

# Chat History Functions
async def save_chat_message(session_id: str, user_question: str, sql_query: str, 
                          ai_response: str, query_results: List[Dict], user_id: str = "default_user"):
    """Save chat message to database"""
    connection = None
    try:
        connection = await get_db_connection()
        async with connection.cursor() as cursor:
            await cursor.execute("""
                INSERT INTO chat_history (session_id, user_question, sql_query, ai_response, 
                                        query_results, user_id, timestamp)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (session_id, user_question, sql_query, ai_response, 
                  json.dumps(query_results, default=str), user_id, datetime.utcnow()))
    except Exception as e:
        logging.error(f"Error saving chat message: {str(e)}")
    finally:
        if connection:
            connection.close()

async def get_chat_history(session_id: str, limit: int = 50) -> List[ChatMessage]:
    """Get chat history for a session"""
    connection = None
    try:
        connection = await get_db_connection()
        async with connection.cursor(aiomysql.DictCursor) as cursor:
            await cursor.execute("""
                SELECT id, session_id, user_question, sql_query, ai_response, timestamp
                FROM chat_history 
                WHERE session_id = %s 
                ORDER BY timestamp DESC 
                LIMIT %s
            """, (session_id, limit))
            results = await cursor.fetchall()
            return [ChatMessage(**dict(row)) for row in results]
    except Exception as e:
        logging.error(f"Error fetching chat history: {str(e)}")
        return []
    finally:
        if connection:
            connection.close()

# File Processing Functions
def process_csv_data(file_content: bytes, table_name: str) -> List[Dict[str, Any]]:
    """Process uploaded CSV file and return formatted data"""
    try:
        # Read CSV
        df = pd.read_csv(io.StringIO(file_content.decode('utf-8')))
        
        # Clean column names (remove spaces, make lowercase)
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # Convert DataFrame to list of dictionaries
        data = df.to_dict('records')
        
        # Process based on table type
        processed_data = []
        for row in data:
            if table_name == "ad_sales":
                processed_row = {
                    'date': pd.to_datetime(row.get('date', datetime.now())).date(),
                    'product_id': str(row.get('product_id', '')),
                    'product_name': str(row.get('product_name', '')),
                    'ad_spend': float(row.get('ad_spend', 0)),
                    'clicks': int(row.get('clicks', 0)),
                    'impressions': int(row.get('impressions', 0)),
                    'ad_sales': float(row.get('ad_sales', 0)),
                    'cpc': float(row.get('cpc', 0)),
                    'cpm': float(row.get('cpm', 0)),
                    'ctr': float(row.get('ctr', 0)),
                    'acos': float(row.get('acos', 0)),
                    'roas': float(row.get('roas', 0))
                }
            elif table_name == "total_sales":
                processed_row = {
                    'date': pd.to_datetime(row.get('date', datetime.now())).date(),
                    'product_id': str(row.get('product_id', '')),
                    'product_name': str(row.get('product_name', '')),
                    'total_sales': float(row.get('total_sales', 0)),
                    'units_sold': int(row.get('units_sold', 0)),
                    'price_per_unit': float(row.get('price_per_unit', 0)),
                    'total_revenue': float(row.get('total_revenue', 0)),
                    'organic_sales': float(row.get('organic_sales', 0)),
                    'sessions': int(row.get('sessions', 0)),
                    'conversion_rate': float(row.get('conversion_rate', 0))
                }
            elif table_name == "product_eligibility":
                processed_row = {
                    'product_id': str(row.get('product_id', '')),
                    'product_name': str(row.get('product_name', '')),
                    'category': str(row.get('category', '')),
                    'brand': str(row.get('brand', '')),
                    'is_eligible_for_ads': bool(row.get('is_eligible_for_ads', True)),
                    'is_active': bool(row.get('is_active', True))
                }
            processed_data.append(processed_row)
            
        return processed_data
        
    except Exception as e:
        logging.error(f"Error processing CSV data: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error processing CSV: {str(e)}")

# Database initialization
async def init_database():
    """Initialize database and create tables"""
    try:
        # Create database if not exists
        connection = await aiomysql.connect(
            host=MYSQL_HOST,
            port=MYSQL_PORT,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            autocommit=True
        )
        
        async with connection.cursor() as cursor:
            await cursor.execute(f"CREATE DATABASE IF NOT EXISTS {MYSQL_DATABASE}")
        connection.close()
        
        # Create tables
        from sqlalchemy import create_engine
        sync_engine = create_engine(f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}")
        Base.metadata.create_all(sync_engine)
        
        logging.info("Database initialized successfully")
        
        # Setup AI processor
        await ai_processor.setup_gemini()
        logging.info("Gemini AI initialized successfully")
        
    except Exception as e:
        logging.error(f"Database initialization failed: {str(e)}")
        raise

# API Routes
@api_router.get("/")
async def root():
    return {"message": "E-commerce Data Query AI Agent v2.0 is running!", "features": ["File Upload", "Chat History", "AI Queries"]}

@api_router.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process natural language query and return results"""
    try:
        # Generate SQL from natural language
        sql_query = await ai_processor.generate_sql(request.question)
        
        # Execute the query
        results = await ai_processor.execute_query(sql_query)
        
        # Format human-readable answer
        human_answer = await ai_processor.format_human_readable(request.question, results)
        
        # Save to chat history
        await save_chat_message(
            session_id=request.session_id,
            user_question=request.question,
            sql_query=sql_query,
            ai_response=human_answer,
            query_results=results,
            user_id=request.user_id
        )
        
        return QueryResponse(
            question=request.question,
            sql_query=sql_query,
            results=results,
            human_readable_answer=human_answer,
            session_id=request.session_id
        )
        
    except Exception as e:
        logging.error(f"Query processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/chat-history/{session_id}")
async def get_session_chat_history(session_id: str, limit: int = 50):
    """Get chat history for a specific session"""
    try:
        history = await get_chat_history(session_id, limit)
        return {"session_id": session_id, "messages": history, "count": len(history)}
    except Exception as e:
        logging.error(f"Error fetching chat history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/upload-csv", response_model=UploadResponse)
async def upload_csv_file(
    file: UploadFile = File(...),
    table_name: str = Form(...)
):
    """Upload CSV file and import data to specified table"""
    
    # Validate table name
    valid_tables = ['ad_sales', 'total_sales', 'product_eligibility']
    if table_name not in valid_tables:
        raise HTTPException(status_code=400, detail=f"Invalid table name. Must be one of: {valid_tables}")
    
    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV file")
    
    try:
        # Read file content
        file_content = await file.read()
        
        # Process CSV data
        processed_data = process_csv_data(file_content, table_name)
        
        # Insert data into database
        connection = await get_db_connection()
        records_inserted = 0
        
        try:
            async with connection.cursor() as cursor:
                # Clear existing data (optional - you might want to append instead)
                await cursor.execute(f"DELETE FROM {table_name}")
                
                # Insert new data
                if table_name == "ad_sales":
                    for row in processed_data:
                        await cursor.execute("""
                            INSERT INTO ad_sales (date, product_id, product_name, ad_spend, clicks, 
                                                impressions, ad_sales, cpc, cpm, ctr, acos, roas)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """, tuple(row.values()))
                        records_inserted += 1
                        
                elif table_name == "total_sales":
                    for row in processed_data:
                        await cursor.execute("""
                            INSERT INTO total_sales (date, product_id, product_name, total_sales, 
                                                   units_sold, price_per_unit, total_revenue, 
                                                   organic_sales, sessions, conversion_rate)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """, tuple(row.values()))
                        records_inserted += 1
                        
                elif table_name == "product_eligibility":
                    for row in processed_data:
                        await cursor.execute("""
                            INSERT INTO product_eligibility (product_id, product_name, category, 
                                                           brand, is_eligible_for_ads, is_active)
                            VALUES (%s, %s, %s, %s, %s, %s)
                            ON DUPLICATE KEY UPDATE
                            product_name=VALUES(product_name), category=VALUES(category), 
                            brand=VALUES(brand), is_eligible_for_ads=VALUES(is_eligible_for_ads), 
                            is_active=VALUES(is_active)
                        """, tuple(row.values()))
                        records_inserted += 1
                
                # Log upload
                await cursor.execute("""
                    INSERT INTO data_upload_logs (filename, table_name, records_count, status)
                    VALUES (%s, %s, %s, %s)
                """, (file.filename, table_name, records_inserted, "success"))
        
        finally:
            connection.close()
        
        return UploadResponse(
            message=f"Successfully uploaded {records_inserted} records to {table_name}",
            records_uploaded=records_inserted,
            filename=file.filename,
            table_name=table_name
        )
        
    except Exception as e:
        logging.error(f"CSV upload failed: {str(e)}")
        
        # Log failed upload
        try:
            connection = await get_db_connection()
            async with connection.cursor() as cursor:
                await cursor.execute("""
                    INSERT INTO data_upload_logs (filename, table_name, records_count, status, error_message)
                    VALUES (%s, %s, %s, %s, %s)
                """, (file.filename, table_name, 0, "failed", str(e)))
            connection.close()
        except:
            pass
            
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@api_router.get("/upload-logs")
async def get_upload_logs(limit: int = 20):
    """Get recent upload logs"""
    connection = None
    try:
        connection = await get_db_connection()
        async with connection.cursor(aiomysql.DictCursor) as cursor:
            await cursor.execute("""
                SELECT filename, table_name, records_count, upload_timestamp, status, error_message
                FROM data_upload_logs 
                ORDER BY upload_timestamp DESC 
                LIMIT %s
            """, (limit,))
            results = await cursor.fetchall()
            return {"logs": [dict(row) for row in results]}
    except Exception as e:
        logging.error(f"Error fetching upload logs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if connection:
            connection.close()

@api_router.post("/upload-data")
async def upload_data(request: DataUploadRequest):
    """Upload sample data to database tables (legacy endpoint)"""
    connection = None
    try:
        connection = await get_db_connection()
        
        if request.table_name == "ad_sales":
            async with connection.cursor() as cursor:
                for row in request.data:
                    await cursor.execute("""
                        INSERT INTO ad_sales (date, product_id, product_name, ad_spend, clicks, impressions, ad_sales, cpc, cpm, ctr, acos, roas)
                        VALUES (%(date)s, %(product_id)s, %(product_name)s, %(ad_spend)s, %(clicks)s, %(impressions)s, 
                               %(ad_sales)s, %(cpc)s, %(cpm)s, %(ctr)s, %(acos)s, %(roas)s)
                    """, row)
                    
        elif request.table_name == "total_sales":
            async with connection.cursor() as cursor:
                for row in request.data:
                    await cursor.execute("""
                        INSERT INTO total_sales (date, product_id, product_name, total_sales, units_sold, price_per_unit, 
                                               total_revenue, organic_sales, sessions, conversion_rate)
                        VALUES (%(date)s, %(product_id)s, %(product_name)s, %(total_sales)s, %(units_sold)s, 
                               %(price_per_unit)s, %(total_revenue)s, %(organic_sales)s, %(sessions)s, %(conversion_rate)s)
                    """, row)
                    
        elif request.table_name == "product_eligibility":
            async with connection.cursor() as cursor:
                for row in request.data:
                    await cursor.execute("""
                        INSERT INTO product_eligibility (product_id, product_name, category, brand, is_eligible_for_ads, is_active)
                        VALUES (%(product_id)s, %(product_name)s, %(category)s, %(brand)s, %(is_eligible_for_ads)s, %(is_active)s)
                        ON DUPLICATE KEY UPDATE
                        product_name=VALUES(product_name), category=VALUES(category), brand=VALUES(brand),
                        is_eligible_for_ads=VALUES(is_eligible_for_ads), is_active=VALUES(is_active)
                    """, row)
        
        return {"message": f"Successfully uploaded {len(request.data)} records to {request.table_name}"}
        
    except Exception as e:
        logging.error(f"Data upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if connection:
            connection.close()

@api_router.get("/sample-data")
async def create_sample_data():
    """Create sample e-commerce data for testing"""
    try:
        connection = await get_db_connection()
        
        # Sample data
        sample_ad_sales = [
            {
                'date': '2024-01-15', 'product_id': 'PROD001', 'product_name': 'Wireless Bluetooth Headphones',
                'ad_spend': 150.50, 'clicks': 320, 'impressions': 8500, 'ad_sales': 850.75,
                'cpc': 0.47, 'cpm': 17.71, 'ctr': 3.76, 'acos': 17.70, 'roas': 5.65
            },
            {
                'date': '2024-01-16', 'product_id': 'PROD002', 'product_name': 'Gaming Mouse RGB',
                'ad_spend': 89.25, 'clicks': 180, 'impressions': 4200, 'ad_sales': 445.60,
                'cpc': 0.50, 'cpm': 21.25, 'ctr': 4.29, 'acos': 20.03, 'roas': 4.99
            }
        ]
        
        sample_total_sales = [
            {
                'date': '2024-01-15', 'product_id': 'PROD001', 'product_name': 'Wireless Bluetooth Headphones',
                'total_sales': 1250.80, 'units_sold': 25, 'price_per_unit': 50.03, 'total_revenue': 1250.80,
                'organic_sales': 400.05, 'sessions': 890, 'conversion_rate': 2.81
            },
            {
                'date': '2024-01-16', 'product_id': 'PROD002', 'product_name': 'Gaming Mouse RGB',
                'total_sales': 780.45, 'units_sold': 15, 'price_per_unit': 52.03, 'total_revenue': 780.45,
                'organic_sales': 334.85, 'sessions': 560, 'conversion_rate': 2.68
            }
        ]
        
        sample_eligibility = [
            {
                'product_id': 'PROD001', 'product_name': 'Wireless Bluetooth Headphones',
                'category': 'Electronics', 'brand': 'TechBrand', 'is_eligible_for_ads': True, 'is_active': True
            },
            {
                'product_id': 'PROD002', 'product_name': 'Gaming Mouse RGB',
                'category': 'Gaming', 'brand': 'GamePro', 'is_eligible_for_ads': True, 'is_active': True
            }
        ]
        
        # Insert sample data
        async with connection.cursor() as cursor:
            # Clear existing data
            await cursor.execute("DELETE FROM ad_sales")
            await cursor.execute("DELETE FROM total_sales")
            await cursor.execute("DELETE FROM product_eligibility")
            
            # Insert ad sales
            for row in sample_ad_sales:
                await cursor.execute("""
                    INSERT INTO ad_sales (date, product_id, product_name, ad_spend, clicks, impressions, ad_sales, cpc, cpm, ctr, acos, roas)
                    VALUES (%(date)s, %(product_id)s, %(product_name)s, %(ad_spend)s, %(clicks)s, %(impressions)s, 
                           %(ad_sales)s, %(cpc)s, %(cpm)s, %(ctr)s, %(acos)s, %(roas)s)
                """, row)
            
            # Insert total sales
            for row in sample_total_sales:
                await cursor.execute("""
                    INSERT INTO total_sales (date, product_id, product_name, total_sales, units_sold, price_per_unit, 
                                           total_revenue, organic_sales, sessions, conversion_rate)
                    VALUES (%(date)s, %(product_id)s, %(product_name)s, %(total_sales)s, %(units_sold)s, 
                           %(price_per_unit)s, %(total_revenue)s, %(organic_sales)s, %(sessions)s, %(conversion_rate)s)
                """, row)
            
            # Insert eligibility
            for row in sample_eligibility:
                await cursor.execute("""
                    INSERT INTO product_eligibility (product_id, product_name, category, brand, is_eligible_for_ads, is_active)
                    VALUES (%(product_id)s, %(product_name)s, %(category)s, %(brand)s, %(is_eligible_for_ads)s, %(is_active)s)
                """, row)
        
        connection.close()
        return {"message": "Sample data created successfully"}
        
    except Exception as e:
        logging.error(f"Sample data creation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Include router and middleware
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Startup event
@app.on_event("startup")
async def startup_event():
    await init_database()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)