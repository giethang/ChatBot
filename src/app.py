# app.py — Chatbot with Conversational Memory for SQL Analysis
import os
import re
import pandas as pd
import streamlit as st
from typing import Dict, Any, List, Optional
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from urllib.parse import quote_plus
from dotenv import load_dotenv
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

# ================================================================
# 0) CONFIG
# ================================================================
st.set_page_config(page_title="Sales Analysis Chatbot", page_icon="🤖", layout="wide")
load_dotenv()

SERVER_HOST = os.getenv("MSSQL_HOST", "10.1.1.4")
SERVER_PORT = os.getenv("MSSQL_PORT", "1433")
SQLSERVER_DRIVER = os.getenv("MSSQL_ODBC_DRIVER", "ODBC Driver 17 for SQL Server")
SQL_USER = os.getenv("MSSQL_USER", "gthang")
SQL_PASSWORD = os.getenv("MSSQL_PASSWORD", "")
ALLOWED_DATABASES = ["MrDairyNovus"]

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# ================================================================
# 1) DB CONNECTION HELPERS
# ================================================================
def make_engine(database: str) -> Engine:
    odbc_str = (
        f"Driver={SQLSERVER_DRIVER};"
        f"Server={SERVER_HOST},{SERVER_PORT};"
        f"Database={database};"
        f"Uid={SQL_USER};"
        f"Pwd={SQL_PASSWORD};"
        f"TrustServerCertificate=yes;"
        f"Encrypt=no;"
        f"ApplicationIntent=ReadOnly;"
        f"Connection Timeout=10;"
    )
    connect_str = quote_plus(odbc_str)
    return create_engine(f"mssql+pyodbc:///?odbc_connect={connect_str}", pool_pre_ping=True, future=True, echo=False)

@st.cache_resource(show_spinner=False)
def get_engine(database: str) -> Engine:
    return make_engine(database)

# ================================================================
# 2) LLM HELPERS
# ================================================================
def get_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=GROQ_API_KEY,
        temperature=0,
        max_tokens=800,
        streaming=False,
    )

# ================================================================
# 3) PROMPTS
# ================================================================

# Router prompt: Decide if we need new data or can analyze existing results
ROUTER_PROMPT = ChatPromptTemplate.from_template("""
You are a routing assistant. Determine if the user's question requires fetching NEW data from the database, or if it can be answered by ANALYZING the existing results.

Current conversation context:
{history}

Current question: {question}

Available data summary:
{data_summary}

Rules:
- If this is the FIRST question in the conversation, respond with: FETCH_NEW_DATA
- If the user asks for DIFFERENT date ranges, products, customers, or dimensions → FETCH_NEW_DATA
- If the user asks to analyze, filter, calculate, or get details about EXISTING results → ANALYZE_EXISTING
- Keywords that suggest ANALYZE_EXISTING: "what is the total", "show me", "which one", "how many", "filter", "only show", "between", "greater than"
- Keywords that suggest FETCH_NEW_DATA: "change to", "instead show", "different period", "new query"

Respond with ONLY one of these two words:
FETCH_NEW_DATA
ANALYZE_EXISTING
""")

# SQL Generation prompt with detailed filtering instructions
EXEC_PROMPT = ChatPromptTemplate.from_template("""
You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
Based on the table schema below, write a SQL query that would answer the user's question. Take the conversation history into account.
You write exactly ONE SQL command for Microsoft SQL Server.

Task: Given the user's question, output ONE line:
EXEC dbo.usp_OE_SalesAnalysis @StartDate='YYYY-MM-DD', @EndDate='YYYY-MM-DD', @IncludeSales=1|0, @IncludeCredits=1|0, @IncludeCrates=1|0, @ReportBy='...', @ReportValue='...', @Offset=int, @Fetch=int

Available Parameters:
FILTERS (use to narrow down data):
  @BillToId='...' - Filter by specific bill-to customer ID
  @ShipToId='...' - Filter by specific ship-to customer ID
  @AccountClass='...' - Filter by account classification
  @AccountRank='...' - Filter by account rank
  @SalesRep='...' - Filter by sales representative
  @CSR='...' - Filter by customer service rep
  @ProductId='...' - Filter by specific product (e.g., '1% 4L BAG')
  @ProductClass='...' - Filter by product category

TRANSACTION TYPES:
  @IncludeSales=1|0 - Include sales transactions
  @IncludeCredits=1|0 - Include credit/return transactions
  @IncludeCrates=1|0 - Include crate transactions

REPORT BY (determines grouping/what shows in rows):
  'Firm' - Group by customer/firm (shows firm names)
  'Product' - Group by product (shows product IDs/names)
  'ProductClass' - Group by product category
  'AccountClass' - Group by account classification
  'SalesRep' - Group by sales representative
  'CSR' - Group by customer service rep

REPORT VALUE (what metric to show in period columns):
  'Sales' - Show sales amounts
  'Cost' - Show cost amounts
  'Profit' - Show profit amounts
  'Units' - Show unit quantities

Rules for interpreting user intent:
1. FILTERING: When user mentions a specific product, firm, or filter → use the appropriate @Filter parameter
   Example: "show me 1% 4L BAG" → @ProductId='1% 4L BAG'
   Example: "sales for Costco" → @BillToId='Costco' (or appropriate ID)

2. REPORT BY: Choose based on what user wants to see in rows
   - "show products" → @ReportBy='Product'
   - "show firms" or "show customers" → @ReportBy='Firm'
   - "by sales rep" → @ReportBy='SalesRep'
   - "with their firm" or "and their customers" → @ReportBy='Product' (product in rows, firm info comes from filter context)

3. COMBINING FILTERS: 
   - "1% 4L BAG with their firms" → @ProductId='1% 4L BAG', @ReportBy='Firm'
   - "products for ABC Company" → @BillToId='ABC Company', @ReportBy='Product'

4. REPORT VALUE: Choose based on what metric user asks for
   - "sales" → @ReportValue='Sales'
   - "profit" → @ReportValue='Profit'
   - "units" or "quantity" → @ReportValue='Units'
   - "cost" → @ReportValue='Cost'

5. DATES:
   - If user gives no dates, use last 90 days: calculate @StartDate and @EndDate
   - If user says "last month", "this year", etc. calculate appropriate dates
   - Always use YYYY-MM-DD format

6. DEFAULTS if unclear:
   - @IncludeSales=1, @IncludeCredits=1, @IncludeCrates=1
   - @ReportValue='Sales'
   - @Offset=0, @Fetch=100000

Output Requirements:
- Output ONLY the EXEC line. No explanation, no code fences, no extra text.
- All string parameters must be single-quoted
- All dates must be 'YYYY-MM-DD'
- Bits must be 0 or 1

Conversation:
{history}

User question: {question}
""")

# Analysis prompt: Analyze existing data
ANALYSIS_PROMPT = ChatPromptTemplate.from_template("""
You are a data analyst. The user has asked a question about existing data that's already been fetched.

Conversation history:
{history}

Current results (showing first 50 rows):
{data_preview}

Data columns available: {columns}
Total rows in dataset: {total_rows}

User question: {question}

Task: Answer the user's question by analyzing the data shown above. Provide:
1. A direct answer to their question
2. Relevant calculations or summaries
3. Any notable insights

If the question requires filtering or calculations:
- Describe what rows/values meet the criteria
- Show specific numbers and totals
- Reference specific column values when relevant

Keep your response concise and focused on answering the question.
""")

# ================================================================
# 4) VALIDATION AND EXECUTION
# ================================================================
_ALLOWED_PARAMS = {
    "StartDate","EndDate","BillToId","ShipToId","AccountClass","AccountRank",
    "SalesRep","CSR","ProductId","ProductClass",
    "IncludeSales","IncludeCredits","IncludeCrates",
    "ReportBy","ReportValue","Offset","Fetch"
}
_ENUM_REPORTBY = {"Firm","AccountClass","SalesRep","CSR","Product","ProductClass"}
_ENUM_REPORTVALUE = {"Sales","Cost","Profit","Units"}

def _validate_exec_sql(exec_sql: str) -> str:
    sql = exec_sql.strip().rstrip(";")
    if not sql.lower().startswith("exec dbo.usp_oe_salesanalysis"):
        raise ValueError("Only EXEC dbo.usp_OE_SalesAnalysis is allowed.")
    pairs = re.findall(r"@([A-Za-z]+)\s*=\s*([^,]+)(?=,|$)", sql)
    if not pairs:
        return "EXEC dbo.usp_OE_SalesAnalysis"
    seen = {}
    for name, raw in pairs:
        if name not in _ALLOWED_PARAMS:
            raise ValueError(f"Unknown parameter: {name}")
        val = raw.strip()
        if name in {"IncludeSales","IncludeCredits","IncludeCrates"}:
            if val not in {"0","1"}:
                raise ValueError(f"{name} must be 0 or 1.")
        elif name in {"Offset","Fetch"}:
            if not re.fullmatch(r"\d+", val):
                raise ValueError(f"{name} must be integer.")
        elif name in {"StartDate","EndDate"}:
            if not re.fullmatch(r"'[0-9]{4}-[0-9]{2}-[0-9]{2}'", val):
                raise ValueError(f"{name} must be 'YYYY-MM-DD'.")
        elif name == "ReportBy":
            m = re.fullmatch(r"'([A-Za-z]+)'", val)
            if not m or m.group(1) not in _ENUM_REPORTBY:
                raise ValueError("ReportBy invalid.")
        elif name == "ReportValue":
            m = re.fullmatch(r"'([A-Za-z]+)'", val)
            if not m or m.group(1) not in _ENUM_REPORTVALUE:
                raise ValueError("ReportValue invalid.")
        else:
            if not (val.startswith("'") and val.endswith("'")):
                raise ValueError(f"{name} must be a single-quoted string.")
            if ";" in val or "--" in val or "/*" in val:
                raise ValueError("Invalid characters in string parameter.")
        seen[name] = val
    order = [
        "StartDate","EndDate","BillToId","ShipToId","AccountClass","AccountRank",
        "SalesRep","CSR","ProductId","ProductClass",
        "IncludeSales","IncludeCredits","IncludeCrates",
        "ReportBy","ReportValue","Offset","Fetch"
    ]
    assigns = [f"@{k}={seen[k]}" for k in order if k in seen]
    return "EXEC dbo.usp_OE_SalesAnalysis" + ((" " + ", ".join(assigns)) if assigns else "")

def run_exec_sql(engine: Engine, exec_sql: str) -> pd.DataFrame:
    safe_exec = _validate_exec_sql(exec_sql)
    final_sql = "SET NOCOUNT ON; " + safe_exec
    with engine.connect() as c:
        result = c.exec_driver_sql(final_sql)
        rows = result.fetchall()
        cols = result.keys()
        return pd.DataFrame(rows, columns=cols)

# ================================================================
# 5) HELPER FUNCTIONS
# ================================================================
PERIOD_COLS = [f"Period{i}" for i in range(1, 13)]

def convert_periods_to_dates(df: pd.DataFrame, end_date: str = None) -> pd.DataFrame:
    """
    Convert Period1, Period2, ... columns to actual date columns.
    Period 12 is always the END month, and periods count backwards from there.
    
    Args:
        df: DataFrame with Period columns
        end_date: Ending date in 'YYYY-MM-DD' format. If None, uses current date
    
    Returns:
        DataFrame with Period columns renamed to actual month-year dates
    """
    if df is None or df.empty:
        return df
    
    # Find all Period columns in the dataframe
    period_cols = [col for col in df.columns if col.startswith('Period') and col[6:].isdigit()]
    
    if not period_cols:
        return df
    
    # Determine end date (Period 12 is the end month)
    if end_date is None:
        ref_date = datetime.now()
    else:
        ref_date = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Get the first day of the end month
    end_month = ref_date.replace(day=1)
    
    # Create mapping of Period columns to date strings
    rename_map = {}
    for col in period_cols:
        period_num = int(col.replace('Period', ''))
        # Period 12 = end month, Period 11 = 1 month before, etc.
        # So: months_back = 12 - period_num
        months_back = 12 - period_num
        date_for_period = end_month - relativedelta(months=months_back)
        
        # Format as "Mon YYYY" (e.g., "Oct 2025", "Sep 2025")
        date_str = date_for_period.strftime('%b %Y')
        rename_map[col] = date_str
    
    # Rename columns
    df_renamed = df.rename(columns=rename_map)
    
    return df_renamed

def add_row_totals(df, value_col_name="TotalAcrossPeriods"):
    """Add a total column summing across period/date columns"""
    if df is None or df.empty: 
        return df
    
    # Look for both Period columns and date-formatted columns (YYYY-MM pattern)
    period_cols = [c for c in df.columns if c.startswith('Period') and c[6:].isdigit()]
    date_cols = [c for c in df.columns if re.match(r'[A-Z][a-z]{2} \d{4}', c)]
    
    cols_to_sum = period_cols + date_cols
    
    out = df.copy()
    if cols_to_sum:
        out[value_col_name] = out[cols_to_sum].fillna(0).sum(axis=1)
    else:
        out[value_col_name] = 0
    
    return out

def format_dataframe_for_display(df: pd.DataFrame, exec_sql: str = None) -> pd.DataFrame:
    """
    Format the dataframe for better display based on the query context.
    Reorder columns to show the most relevant information first.
    """
    if df is None or df.empty:
        return df
    
    display_df = df.copy()
    
    # Extract what we're reporting by from the query
    report_by = None
    if exec_sql:
        match = re.search(r"@ReportBy='([^']+)'", exec_sql)
        if match:
            report_by = match.group(1)
    
    # Define preferred column order based on report type
    id_cols = []
    name_cols = []
    metric_cols = []
    period_cols = []
    
    for col in display_df.columns:
        col_lower = col.lower()
        # ID columns
        if 'id' in col_lower and col != 'TotalAcrossPeriods':
            id_cols.append(col)
        # Name/description columns
        elif 'name' in col_lower or 'description' in col_lower or col in ['Firm', 'FirmName', 'Product', 'ProductName', 'SalesRep', 'CSR', 'AccountClass']:
            name_cols.append(col)
        # Date/period columns (check for both "Period" and month formats like "Oct 2025")
        elif col.startswith('Period') or re.match(r'[A-Z][a-z]{2} \d{4}', col):
            period_cols.append(col)
        # Total column
        elif col == 'TotalAcrossPeriods':
            metric_cols.append(col)
        # Other numeric columns
        elif display_df[col].dtype in ['int64', 'float64']:
            metric_cols.append(col)
    
    # Sort period columns properly
    if period_cols:
        # If they're date formatted (e.g., "Oct 2025"), sort chronologically
        if re.match(r'[A-Z][a-z]{2} \d{4}', period_cols[0]):
            period_cols = sorted(period_cols, key=lambda x: datetime.strptime(x, '%b %Y'))
        # If they're Period1, Period2, etc., sort numerically
        else:
            period_cols = sorted(period_cols, key=lambda x: int(x.replace('Period', '')))
    
    # Build the preferred column order
    ordered_cols = id_cols + name_cols + period_cols + metric_cols
    
    # Add any remaining columns that weren't categorized
    remaining = [col for col in display_df.columns if col not in ordered_cols]
    ordered_cols.extend(remaining)
    
    # Reorder the dataframe
    display_df = display_df[ordered_cols]
    
    return display_df

def get_data_summary(df: pd.DataFrame) -> str:
    """Create a summary of the dataframe for the router"""
    if df is None or df.empty:
        return "No data currently loaded."
    
    summary_parts = [
        f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns.",
        f"Columns: {', '.join(df.columns.tolist()[:10])}{'...' if len(df.columns) > 10 else ''}",
    ]
    
    # Add some basic stats
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        sample_col = numeric_cols[0]
        summary_parts.append(f"Sample numeric column '{sample_col}': min={df[sample_col].min():.2f}, max={df[sample_col].max():.2f}")
    
    return " ".join(summary_parts)

def format_history_for_llm(history: List[Dict], max_messages: int = 10) -> str:
    """Format recent chat history for LLM context"""
    recent = history[-max_messages:]
    return "\n".join([f"{m['role']}: {m['content']}" for m in recent])

def extract_end_date_from_query(exec_sql: str) -> Optional[str]:
    """Extract the EndDate parameter from the EXEC query"""
    match = re.search(r"@EndDate='(\d{4}-\d{2}-\d{2})'", exec_sql)
    if match:
        return match.group(1)
    return None

# ================================================================
# 6) STREAMLIT UI
# ================================================================
st.title("🤖 Sales Analysis Chatbot")
st.caption("Ask questions about sales data. I'll remember our conversation and help you drill down into results!")

# Initialize session state
if "history" not in st.session_state:
    st.session_state["history"] = []
if "current_data" not in st.session_state:
    st.session_state["current_data"] = None
if "last_query" not in st.session_state:
    st.session_state["last_query"] = None

# Sidebar with current data info and date conversion toggle
with st.sidebar:
    st.header("📊 Current Data")
    
    # Add toggle for date conversion
    use_dates = st.checkbox("Convert Periods to Dates", value=True, 
                           help="Show actual dates instead of Period1, Period2, etc.")
    
    if st.session_state["current_data"] is not None:
        df = st.session_state["current_data"]
        st.metric("Rows", len(df))
        st.metric("Columns", len(df.columns))
        if st.session_state["last_query"]:
            st.code(st.session_state["last_query"], language="sql")
        if st.button("Clear Data & Start Fresh"):
            st.session_state["current_data"] = None
            st.session_state["last_query"] = None
            st.session_state["history"] = []
            st.rerun()
    else:
        st.info("No data loaded yet. Ask a question to fetch data!")

# Display chat history
for msg in st.session_state["history"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Show dataframes if they exist in the message
        if "dataframe" in msg:
            display_df = msg["dataframe"].copy()
            
            # Apply date conversion if enabled
            if use_dates and "query" in msg:
                end_date = extract_end_date_from_query(msg["query"])
                display_df = convert_periods_to_dates(display_df, end_date)
            
            st.dataframe(display_df, use_container_width=True)

# Chat input
user_q = st.chat_input("Ask me a question about sales")
if user_q:
    # Add user message to history
    st.session_state["history"].append({"role": "user", "content": user_q})
    
    with st.chat_message("user"):
        st.markdown(user_q)
    
    with st.chat_message("assistant"):
        try:
            # Step 1: Route the question
            router_chain = ROUTER_PROMPT | get_llm() | StrOutputParser()
            
            data_summary = get_data_summary(st.session_state["current_data"])
            history_str = format_history_for_llm(st.session_state["history"])
            
            routing_decision = router_chain.invoke({
                "history": history_str,
                "question": user_q,
                "data_summary": data_summary
            }).strip()
            
            # Debug: show routing decision
            with st.expander("🔍 Routing Decision", expanded=False):
                st.write(f"Decision: **{routing_decision}**")
            
            if "FETCH_NEW_DATA" in routing_decision or st.session_state["current_data"] is None:
                # Generate and execute SQL
                st.info("📊 Fetching new data from database...")
                
                exec_chain = EXEC_PROMPT | get_llm() | StrOutputParser()
                exec_sql = exec_chain.invoke({
                    "history": history_str,
                    "question": user_q
                }).strip()
                
                st.code(exec_sql, language="sql")
                
                # Execute query
                eng = get_engine(ALLOWED_DATABASES[0])
                df = run_exec_sql(eng, exec_sql)
                
                # Extract end date for period conversion
                end_date = extract_end_date_from_query(exec_sql)
                
                # Convert periods to dates if enabled
                if use_dates:
                    df = convert_periods_to_dates(df, end_date)
                
                # Add totals (works with both period and date columns)
                df = add_row_totals(df, "TotalAcrossPeriods")
                
                # Format dataframe for better display
                df = format_dataframe_for_display(df, exec_sql)
                
                # Store in session state
                st.session_state["current_data"] = df
                st.session_state["last_query"] = exec_sql
                
                # Display results
                st.success(f"✅ Loaded {len(df)} rows")
                st.dataframe(df.head(50), use_container_width=True)
                
                # Download button
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Full Results (CSV)",
                    data=csv,
                    file_name="sales_analysis.csv",
                    mime="text/csv"
                )
                
                response_msg = f"I've fetched {len(df)} rows of data based on your filters. You can now ask follow-up questions to analyze these results!"
                st.markdown(response_msg)
                
                # Add to history with query for date conversion
                st.session_state["history"].append({
                    "role": "assistant",
                    "content": response_msg,
                    "dataframe": df.head(50),
                    "query": exec_sql
                })
                
            else:
                # Analyze existing data
                st.info("🔎 Analyzing existing results...")
                
                df = st.session_state["current_data"]
                
                # Apply date conversion for display if enabled
                display_df = df.copy()
                if use_dates and st.session_state["last_query"]:
                    end_date = extract_end_date_from_query(st.session_state["last_query"])
                    display_df = convert_periods_to_dates(display_df, end_date)
                
                analysis_chain = ANALYSIS_PROMPT | get_llm() | StrOutputParser()
                analysis = analysis_chain.invoke({
                    "history": history_str,
                    "question": user_q,
                    "data_preview": display_df.head(50).to_string(),
                    "columns": ", ".join(display_df.columns.tolist()),
                    "total_rows": len(display_df)
                })
                
                st.markdown(analysis)
                
                # Try to show relevant filtered data if applicable
                question_lower = user_q.lower()
                if any(word in question_lower for word in ["show", "display", "list", "what are"]):
                    st.dataframe(display_df.head(20), use_container_width=True)
                
                # Add to history
                st.session_state["history"].append({
                    "role": "assistant",
                    "content": analysis
                })
        
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            st.error(error_msg)
            st.session_state["history"].append({
                "role": "assistant",
                "content": error_msg
            })