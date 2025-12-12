# app.py — Chatbot with Conversational Memory for SQL Analysis (Table-Only, No Stored Proc)
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
    return create_engine(
        f"mssql+pyodbc:///?odbc_connect={connect_str}",
        pool_pre_ping=True,
        future=True,
        echo=False,
    )

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
You are a routing assistant for a SQL analytics chatbot.

Your job is to decide if the user's question requires:
- FETCH_NEW_DATA: run a NEW SQL SELECT query against the database tables
- ANALYZE_EXISTING: answer by analyzing the dataset that is ALREADY loaded in memory

Current conversation context:
{history}

Current question: {question}

Available data summary:
{data_summary}

Rules:
- If this is the FIRST question in the conversation → FETCH_NEW_DATA
- If the user changes the date range, time period, product, customer, or dimension → FETCH_NEW_DATA
- If the user asks to drill down, filter, aggregate, compare, or summarize based on ALREADY DISPLAYED results → ANALYZE_EXISTING
- Keywords that often mean ANALYZE_EXISTING:
  "what is the total", "which one", "how many", "filter", "only show", "between", "greater than", "from the above", "in this table"
- Keywords that often mean FETCH_NEW_DATA:
  "change to", "instead show", "different period", "new query", "for last year", "for September", "for this customer" (if it wasn't in the previous query)

Respond with ONLY one of these two words (no explanation):
FETCH_NEW_DATA
ANALYZE_EXISTING
""")

# SQL prompt: Write a SELECT query
SELECT_PROMPT = ChatPromptTemplate.from_template("""
You are a senior data analyst writing SQL for Microsoft SQL Server.

Your job is to write exactly ONE read-only SQL query (SELECT or SELECT with CTEs) that directly answers the user’s question.
You must not use EXEC, stored procedures, INSERT, UPDATE, DELETE, DROP, or any write operations.

===============================================================================
AVAILABLE TABLES (MrDairyNovus)
===============================================================================

dbo.Sales_Order
  (order_key, order_no, order_date, invoice_date,
   order_type, order_status, order_stage, order_source,
   is_credited, is_a_credit, applied_to, applied_orders,
   bill_to_key, bill_to_id, bill_to_name,
   ship_to_key, ship_to_id, ship_to_name,
   customer_service_rep, sales_rep_id,
   sub_total, discount_percent, discount_amount,
   surcharge_amount, shipping_amount,
   tax_amount01, tax_amount02, tax_amount03,
   order_total, sales_cost, gross_profit,
   total_weight, total_volume)

dbo.Sales_Order_Line
  (order_line_key, order_key, line, sub_line, line_type,
   product_key, product_id, product_descr, uom,
   qty_ordered, qty_picked, qty_shipped,
   product_price, unit_price,
   sales_amount, sales_cost, gross_profit,
   unit_cost, commission_percent,
   weight, volume)

dbo.Firm
  (firm_key, firm_id, name,
   address, address2, city_key, zip,
   phone, fax, website,
   firm_type, is_active,
   shipvia, route_main,
   CSR_1, CSR_2, CSR_3, CSR_4, CSR_5, CSR_6, CSR_7)

dbo.Firm_Account
  (firm_account_key, firm_key,
   terms_code, status,
   account_class, account_rank,
   sales_rep_id, customer_service_rep,
   credit_limit, price_table,
   min_order_value, inv_discount_percent)

dbo.Product
  (product_key, product_id, description,
   is_active, product_class_id,
   product_type, product_category,
   unit_of_measure, package_quantity, pallet_quantity,
   cost_average, cost_last, cost_currency,
   standard_price)

dbo.Product_Class
  (product_class_key, product_class_id,
   class_description, is_active)

===============================================================================
STANDARD JOINS
===============================================================================
Sales_Order_Line ol  JOIN Sales_Order o   ON ol.order_key = o.order_key
Sales_Order o        JOIN Firm f          ON o.bill_to_key = f.firm_key
Firm f               JOIN Firm_Account a  ON f.firm_key = a.firm_key
Sales_Order_Line ol  JOIN Product p       ON ol.product_id = p.product_id
Product p            JOIN Product_Class pc ON p.product_class_id = pc.product_class_id

===============================================================================
INVOICE FILTERS (ALWAYS APPLY unless user says otherwise)
===============================================================================
o.order_status = 'Invoiced'
(o.is_a_credit = 0 OR o.is_a_credit IS NULL)
AND (o.is_credited = 0 OR o.is_credited IS NULL)

If using Sales_Order_Line (alias ol):
    ol.line_type NOT IN ('Ext-Descr','Int-Descr')

===============================================================================
DATE FILTER LOGIC
===============================================================================
If user specifies explicit start/end dates:
    Use o.invoice_date BETWEEN start AND end (inclusive)

If user specifies a month (e.g., "September 2025"):
    Interpret full month:
        o.invoice_date >= 'YYYY-MM-01'
        AND o.invoice_date <  'YYYY-(MM+1)-01'

If NO dates are given:
    Use last 90 days:
        WHERE o.invoice_date BETWEEN DATEADD(DAY,-90,CAST(GETDATE() AS date))
                                  AND CAST(GETDATE() AS date)

===============================================================================
DATE DIMENSION RULES
===============================================================================
If "monthly", "per month", "each month":
    FORMAT(o.invoice_date, 'yyyy-MM') AS invoice_month
    Include in SELECT and GROUP BY

If "daily", "by date", "each day":
    CAST(o.invoice_date AS date) AS invoice_date
    Include in SELECT and GROUP BY

If combining date + another dimension:
    Include both in SELECT and GROUP BY

===============================================================================
SALES METRIC RULES (CRITICAL)
===============================================================================
DEFAULT RULE:
    For ALL sales-related questions (even if user does NOT mention products):
        → Use line-level metrics from dbo.Sales_Order_Line.

Line-level metrics:
    SUM(ol.sales_amount)  AS total_sales
    SUM(ol.sales_cost)    AS total_cost
    SUM(ol.gross_profit)  AS total_profit
    SUM(ol.qty_shipped)   AS total_units

Why?
    This reflects true product revenue and matches company reporting logic.

ONLY use header-level order_total IF user explicitly requests:
    "invoice total", "grand total", "include shipping",
    "full invoice amount", or mentions order_total.

Header metrics (only when explicitly requested):
    SUM(o.order_total) AS total_sales

===============================================================================
GROUPING RULES
===============================================================================
By customer:
    Use Sales_Order_Line + Sales_Order join
    Group by f.firm_id, f.name  (or o.bill_to_id, o.bill_to_name if firm not required)

By product:
    Group by ol.product_id, p.description

By product class:
    Group by pc.product_class_id, pc.class_description

By sales rep:
    Group by a.customer_service_rep

By date:
    Group by date/month expression used

===============================================================================
TOTAL COLUMN REQUIREMENTS
===============================================================================
If the question includes "total", "sum", "how much", "top", "highest", "least", etc.:
    You MUST include one aggregated metric with a clear alias:
        total_sales, total_cost, total_profit, total_units

If no “by” dimension:
    → Return a single row with only the metric.

===============================================================================
ORDERING RULES
===============================================================================
"top", "most", "highest", etc. → ORDER BY metric DESC
"least", "lowest", etc.       → ORDER BY metric ASC
Use TOP(N) if user requests top N.

===============================================================================
OUTPUT REQUIREMENTS
===============================================================================
- Output ONLY the SQL (no explanation, no comments, no backticks)
- Must be a single valid T-SQL query (CTEs allowed)
- Must answer the user's question exactly

===============================================================================
CONVERSATION HISTORY
===============================================================================
{history}

===============================================================================
USER QUESTION
===============================================================================
{question}
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
# 4) VALIDATION AND EXECUTION (SELECT ONLY, NO EXEC)
# ================================================================
# Forbid dangerous tokens
_FORBID_TOKENS = (
    "--", "/*", "*/",
    "INSERT", "UPDATE", "DELETE", "MERGE",
    "DROP", "ALTER", "TRUNCATE", "EXEC", "CREATE"
)

def _clean_sql_block(sql: str) -> str:
    """
    Remove ``` fences and trailing semicolon from LLM output.
    """
    s = sql.strip()
    # remove opening fence like ``` or ```sql
    s = re.sub(r"^\s*```[a-zA-Z-]*\s*\n?", "", s)
    # remove closing fence ```
    s = re.sub(r"\n?```\s*$", "", s)
    return s.strip().rstrip(";")

def _validate_select_sql(sql: str) -> str:
    """
    Ensure we only run a single SELECT (or WITH ... SELECT) and no dangerous tokens.
    """
    s = _clean_sql_block(sql)
    low = s.lower()
    if not (low.startswith("select") or low.startswith("with ")):
        raise ValueError("LLM must provide a single SELECT (CTEs allowed).")
    up = s.upper()
    if any(tok in up for tok in _FORBID_TOKENS):
        raise ValueError("Unsafe token detected in SQL (DML/DDL/EXEC not allowed).")
    return s

def run_select_sql(engine: Engine, select_sql: str) -> pd.DataFrame:
    safe = _validate_select_sql(select_sql)
    with engine.connect() as c:
        result = c.exec_driver_sql("SET NOCOUNT ON; " + safe)
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

    If the dataframe has no Period* columns, it is returned unchanged.
    """
    if df is None or df.empty:
        return df

    period_cols = [col for col in df.columns if col.startswith('Period') and col[6:].isdigit()]
    if not period_cols:
        return df

    if end_date is None:
        ref_date = datetime.now()
    else:
        ref_date = datetime.strptime(end_date, '%Y-%m-%d')

    end_month = ref_date.replace(day=1)

    rename_map = {}
    for col in period_cols:
        period_num = int(col.replace('Period', ''))
        months_back = 12 - period_num
        date_for_period = end_month - relativedelta(months=months_back)
        date_str = date_for_period.strftime('%b %Y')
        rename_map[col] = date_str

    df_renamed = df.rename(columns=rename_map)
    return df_renamed

def add_row_totals(df, value_col_name="TotalAcrossPeriods"):
    """
    Add a total column summing across period/date columns.
    If there are no such columns, return the dataframe unchanged.
    """
    if df is None or df.empty:
        return df

    period_cols = [c for c in df.columns if c.startswith('Period') and c[6:].isdigit()]
    date_cols = [c for c in df.columns if re.match(r'[A-Z][a-z]{2} \\d{4}', c)]

    cols_to_sum = period_cols + date_cols
    if not cols_to_sum:
        return df

    out = df.copy()
    out[value_col_name] = out[cols_to_sum].fillna(0).sum(axis=1)
    return out

def format_dataframe_for_display(df: pd.DataFrame, sql: str = None) -> pd.DataFrame:
    """
    Format the dataframe for better display based on the query context.
    Attempt to place IDs/names first, then period/date columns, then metrics.
    """
    if df is None or df.empty:
        return df

    display_df = df.copy()

    id_cols = []
    name_cols = []
    metric_cols = []
    period_cols = []

    for col in display_df.columns:
        col_lower = col.lower()
        if 'id' in col_lower and col not in ('TotalAcrossPeriods',):
            id_cols.append(col)
        elif (
            'name' in col_lower
            or 'description' in col_lower
            or col in ['Firm', 'FirmName', 'Product', 'ProductName', 'SalesRep', 'CSR', 'AccountClass']
        ):
            name_cols.append(col)
        elif col.startswith('Period') or re.match(r'[A-Z][a-z]{2} \\d{4}', col):
            period_cols.append(col)
        elif col == 'TotalAcrossPeriods':
            metric_cols.append(col)
        elif display_df[col].dtype in ['int64', 'float64']:
            metric_cols.append(col)

    # sort period/date columns
    if period_cols:
        if re.match(r'[A-Z][a-z]{2} \\d{4}', period_cols[0]):
            period_cols = sorted(period_cols, key=lambda x: datetime.strptime(x, '%b %Y'))
        else:
            period_cols = sorted(period_cols, key=lambda x: int(x.replace('Period', '')))

    ordered_cols = id_cols + name_cols + period_cols + metric_cols
    remaining = [col for col in display_df.columns if col not in ordered_cols]
    ordered_cols.extend(remaining)

    display_df = display_df[ordered_cols]
    return display_df

def get_data_summary(df: pd.DataFrame) -> str:
    """Create a summary of the dataframe for the router."""
    if df is None or df.empty:
        return "No data currently loaded."

    summary_parts = [
        f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns.",
        f"Columns: {', '.join(df.columns.tolist()[:10])}{'...' if len(df.columns) > 10 else ''}",
    ]

    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        sample_col = numeric_cols[0]
        summary_parts.append(
            f"Sample numeric column '{sample_col}': min={df[sample_col].min():.2f}, max={df[sample_col].max():.2f}"
        )

    return " ".join(summary_parts)

def format_history_for_llm(history: List[Dict], max_messages: int = 10) -> str:
    """Format recent chat history for LLM context."""
    recent = history[-max_messages:]
    return "\\n".join([f"{m['role']}: {m['content']}" for m in recent])

# ================================================================
# 6) STREAMLIT UI
# ================================================================
st.image("./VisfutureLogo.png")
st.title(" Sales Analysis Chatbot ")

st.caption("Ask questions about sales data. I’ll generate SELECT queries and remember our conversation so you can drill down on existing results.")

# Initialize session state
if "history" not in st.session_state:
    st.session_state["history"] = []
if "current_data" not in st.session_state:
    st.session_state["current_data"] = None
if "last_query" not in st.session_state:
    st.session_state["last_query"] = None  # now holds the last SELECT

# Sidebar with current data info and date conversion toggle
with st.sidebar:
    st.header(" Current Data")

    use_dates = st.checkbox(
        "Convert Periods to Dates",
        value=True,
        help="If the result has Period1, Period2, etc., show actual months instead."
    )

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

# Display chat history (including any past dataframes)
for msg in st.session_state["history"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "dataframe" in msg:
            display_df = msg["dataframe"].copy()
            if use_dates:
                display_df = convert_periods_to_dates(display_df)
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

            # OVERRIDE: if user asks for metrics not in current columns, force FETCH_NEW_DATA
            if (
                "ANALYZE_EXISTING" in routing_decision
                and st.session_state.get("current_data") is not None
            ):
                df_current = st.session_state["current_data"]
                cols_lower = [c.lower() for c in df_current.columns]
                q_lower = user_q.lower()

                # simple metric keyword checks
                wants_total_sales = ("total sales" in q_lower or "sales amount" in q_lower) and "total_sales" not in cols_lower
                wants_total_units = ("total units" in q_lower or "units sold" in q_lower) and "total_units" not in cols_lower
                wants_total_profit = ("total profit" in q_lower or "gross profit" in q_lower) and "total_profit" not in cols_lower

                if wants_total_sales or wants_total_units or wants_total_profit:
                    routing_decision = "FETCH_NEW_DATA"

            with st.expander(" Routing Decision", expanded=False):
                st.write(f"Decision: **{routing_decision}**")

            if "FETCH_NEW_DATA" in routing_decision or st.session_state["current_data"] is None:
                # Step 2: Generate SELECT SQL
                st.info(" Fetching new data from database (SELECT only)...")

                select_chain = SELECT_PROMPT | get_llm() | StrOutputParser()
                select_sql = select_chain.invoke({
                    "history": history_str,
                    "question": user_q
                }).strip()

                st.code(select_sql, language="sql")

                # Step 3: Execute SELECT
                eng = get_engine(ALLOWED_DATABASES[0])
                df = run_select_sql(eng, select_sql)

                # Optional: convert Period columns to dates (if they exist)
                if use_dates:
                    df = convert_periods_to_dates(df)

                # Optional: add row totals if Period/date columns exist
                df = add_row_totals(df, "TotalAcrossPeriods")

                # Format dataframe for display
                df = format_dataframe_for_display(df, select_sql)

                # Store in session state
                st.session_state["current_data"] = df
                st.session_state["last_query"] = select_sql

                # Display results
                st.success(f" Loaded {len(df)} rows")
                st.dataframe(df.head(50), use_container_width=True)


                response_msg = (
                    f"I've fetched {len(df)} rows of data using a direct SELECT query. "
                    f"You can now ask follow-up questions to analyze these results without re-querying the database."
                )
                st.markdown(response_msg)

                st.session_state["history"].append({
                    "role": "assistant",
                    "content": response_msg,
                    "dataframe": df.head(50),
                })

            else:
                # Step 3b: Analyze existing data in memory
                st.info("🔎 Analyzing existing results (no new SQL query)...")

                df = st.session_state["current_data"]

                display_df = df.copy()
                if use_dates:
                    display_df = convert_periods_to_dates(display_df)

                analysis_chain = ANALYSIS_PROMPT | get_llm() | StrOutputParser()
                analysis = analysis_chain.invoke({
                    "history": history_str,
                    "question": user_q,
                    "data_preview": display_df.head(50).to_string(),
                    "columns": ", ".join(display_df.columns.tolist()),
                    "total_rows": len(display_df)
                })

                st.markdown(analysis)

                # If question sounds like "show/list/what are", show some rows again
                question_lower = user_q.lower()
                if any(word in question_lower for word in ["show", "display", "list", "what are"]):
                    st.dataframe(display_df.head(20), use_container_width=True)

                st.session_state["history"].append({
                    "role": "assistant",
                    "content": analysis
                })

        except Exception as e:
            error_msg = f" Error: {str(e)}"
            st.error(error_msg)
            st.session_state["history"].append({
                "role": "assistant",
                "content": error_msg
            })