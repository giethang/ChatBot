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
from decimal import Decimal
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
# 1B) SCHEMA INTROSPECTION (tables + column data types)
# ================================================================
@st.cache_data(show_spinner=False, ttl=24 * 3600)
def load_schema_df(_engine: Engine, schema: str = "dbo") -> pd.DataFrame:
    sql = """
    SELECT
        c.TABLE_SCHEMA,
        c.TABLE_NAME,
        c.COLUMN_NAME,
        c.ORDINAL_POSITION,
        c.DATA_TYPE,
        c.CHARACTER_MAXIMUM_LENGTH,
        c.NUMERIC_PRECISION,
        c.NUMERIC_SCALE,
        c.DATETIME_PRECISION
    FROM INFORMATION_SCHEMA.COLUMNS c
    WHERE c.TABLE_SCHEMA = :schema
    ORDER BY c.TABLE_NAME, c.ORDINAL_POSITION
    """
    return pd.read_sql(text(sql), _engine, params={"schema": schema})


def _format_sqlserver_type(row: pd.Series) -> str:
    dt = str(row["DATA_TYPE"]).lower()

    if dt in ("varchar", "nvarchar", "char", "nchar", "binary", "varbinary"):
        n = row["CHARACTER_MAXIMUM_LENGTH"]
        if pd.isna(n):
            return dt
        n = int(n)
        return f"{dt}(max)" if n == -1 else f"{dt}({n})"

    if dt in ("decimal", "numeric"):
        p = row["NUMERIC_PRECISION"]
        s = row["NUMERIC_SCALE"]
        if pd.isna(p) or pd.isna(s):
            return dt
        return f"{dt}({int(p)},{int(s)})"

    if dt in ("datetime2", "time"):
        prec = row["DATETIME_PRECISION"]
        if pd.isna(prec):
            return dt
        return f"{dt}({int(prec)})"

    return dt


def build_schema_text(
    schema_df: pd.DataFrame,
    include_only_tables: Optional[List[str]] = None,
    exclude_tables: Optional[List[str]] = None,
    max_cols_per_table: int = 250,
) -> str:
    if schema_df is None or schema_df.empty:
        return "No tables/columns found."

    df = schema_df.copy()

    if exclude_tables:
        ex = set(t.lower() for t in exclude_tables)
        df = df[~df["TABLE_NAME"].str.lower().isin(ex)]

    if include_only_tables:
        inc = set(t.lower() for t in include_only_tables)
        df = df[df["TABLE_NAME"].str.lower().isin(inc)]

    lines = []
    lines.append("===============================================================================")
    lines.append("AVAILABLE TABLES (MrDairyNovus) — columns + SQL Server data types (authoritative)")
    lines.append("===============================================================================")

    for tbl, g in df.groupby("TABLE_NAME", sort=True):
        g = g.sort_values("ORDINAL_POSITION").head(max_cols_per_table)

        col_parts = []
        for _, r in g.iterrows():
            col_name = r["COLUMN_NAME"]
            col_type = _format_sqlserver_type(r)
            col_parts.append(f"{col_name} {col_type}")

        cols_str = ",\n    ".join(col_parts)
        lines.append(f"\ndbo.{tbl}\n  (\n    {cols_str}\n  )")

    return "\n".join(lines)

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

{schema_text}

===============================================================================
COLUMN UNIT HINTS (USE THESE FOR OUTPUT FORMATTING)
===============================================================================

IMPORTANT:
- Return raw numeric values in SQL (NO FORMAT()).
- The UI applies $ / % / commas using these hints.

CURRENCY (show as $):
Sales / invoice money:
- ol.sales_amount, o.sub_total, o.discount_amount, o.surcharge_amount, o.shipping_amount
- o.tax_amount01, o.tax_amount02, o.tax_amount03
- o.order_total, o.sales_cost, ol.sales_cost
- o.gross_profit, ol.gross_profit
- ol.unit_price, ol.product_price, ol.unit_cost
- o.freight_cost, o.tariff_amount, ol.tariff_amount
- o.commission_amount

Customer account money:
- a.credit_limit, a.min_order_value
- a.shipping_charges, a.free_shipping_min_order

Product money:
- p.cost_average, p.cost_last, p.standard_price

Rate / pricing money:
- rv.rate, rdv.rate   (Rate_Value.rate, Rate_Discount_Value.rate)
- ol.price_discount_amount, ol.discount_amount

PERCENT (show as %):
- o.discount_percent
- ol.commission_percent
- ol.price_discount_percent
- ol.tariff_percent
- a.inv_discount_percent
- a.is_fluctuation_skipped (BIT flag, not percent — do NOT format as %)

QUANTITY (show as number with commas, no $):
Line quantities:
- ol.qty_ordered, ol.qty_picked, ol.qty_shipped
Packaging quantities:
- p.package_quantity, p.pallet_quantity
Order handling counts:
- o.pallets, o.packages, o.pieces, o.handling_units
Rate ranges:
- rv.low_range_quantity, rv.high_range_quantity
- rdv.low_range_quantity, rdv.high_range_quantity

WEIGHT:
- o.total_weight, o.weight
- ol.weight
- p.dim_weight, p.package_weight, p.pallet_weight

VOLUME / CUBE:
- o.total_volume, o.cube
- ol.volume
- p.dim_cube, p.package_cube, p.pallet_cube

DIMENSIONS (length/width/height) — treat as numeric, not currency:
- p.dim_height, p.dim_length, p.dim_width
- p.package_height, p.package_length, p.package_width
- p.pallet_height, p.pallet_length, p.pallet_width

===============================================================================
ALIAS CONVENTIONS (MUST USE THESE ALIASES WHEN RETURNING METRICS)
===============================================================================
Currency totals:
- SUM(ol.sales_amount)  AS total_sales
- SUM(ol.sales_cost)    AS total_cost
- SUM(ol.gross_profit)  AS total_profit

Quantity totals:
- SUM(ol.qty_shipped)   AS total_units

Percent metrics:
- AVG(o.discount_percent) AS avg_discount_percent
- AVG(ol.commission_percent) AS avg_commission_percent

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
- NEVER use FORMAT() or convert numeric metrics to strings. Return raw numeric values. Formatting ($, %, commas) is handled by the UI.
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

def infer_unit_from_column_name(col: str) -> str:
    c = col.lower().strip()

    # --- Quantity FIRST (so "total_units" doesn't get caught by "total") ---
    if any(k in c for k in [
        "total_units", "units", "unit", "qty", "quantity", "count", "pieces", "packages", "pallets", "handling_units"
    ]):
        return "quantity"

    # Percent
    if any(k in c for k in ["percent", "pct", "ratio", "rate", "margin"]):
        return "percent"

    # Weight / Volume
    if "weight" in c:
        return "weight"
    if any(k in c for k in ["volume", "cube"]):
        return "volume"

    # Currency (keep "total" OUT of this list)
    if any(k in c for k in ["sales", "revenue", "amount", "cost", "profit", "price", "freight", "tariff", "commission"]):
        return "currency"

    return "number"

def build_column_config(df: pd.DataFrame) -> dict:
    config = {}
    if df is None or df.empty:
        return config

    for col in df.columns:
        col_lower = col.lower()

        # 1) Quantity FIRST (avoid "unit" to not catch unit_price)
        if any(k in col_lower for k in [
            "total_units", "units", "qty", "quantity", "count",
            "pieces", "packages", "pallets", "handling_units"
        ]):
            config[col] = st.column_config.NumberColumn(label=col, format="%.0f")
            continue

        # 2) Percent
        if any(k in col_lower for k in ["percent", "percentage", "pct", "rate", "margin"]):
            config[col] = st.column_config.NumberColumn(label=f"{col} (%)", format="%.2f")
            continue

        # 3) Weight / Volume
        if "weight" in col_lower:
            config[col] = st.column_config.NumberColumn(label=col, format="%.2f")
            continue

        if any(k in col_lower for k in ["volume", "cube"]):
            config[col] = st.column_config.NumberColumn(label=col, format="%.2f")
            continue

        # 4) Currency LAST (NO comma format - prevents sprintf error)
        if any(k in col_lower for k in [
            "sales", "revenue", "amount", "cost", "profit", "price",
            "freight", "tariff", "commission", "credit_limit", "min_order_value"
        ]):
            config[col] = st.column_config.NumberColumn(label=col, format="$%.2f")
            continue

    return config


def coerce_numeric_objects(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert Decimal/object numeric columns to float where possible so
    Streamlit NumberColumn formatting applies.
    """
    if df is None or df.empty:
        return df

    out = df.copy()

    # First pass: convert Decimal cells -> float
    for col in out.columns:
        if out[col].dtype == "object":
            # if the column contains Decimals, convert them
            if out[col].map(lambda x: isinstance(x, Decimal)).any():
                out[col] = out[col].map(lambda x: float(x) if isinstance(x, Decimal) else x)

    # Second pass: try to coerce object columns that are numeric-like strings/values
    for col in out.columns:
        if out[col].dtype == "object":
            coerced = pd.to_numeric(out[col], errors="ignore")
            out[col] = coerced

    return out

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
            st.dataframe(
                display_df,
                column_config=build_column_config(display_df),
                use_container_width=True
            )

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
                # Build schema_text (tables + data types) once per run (cached)
                eng = get_engine(ALLOWED_DATABASES[0])
                schema_df = load_schema_df(eng, schema="dbo")

                TABLE_WHITELIST = [
                    "Sales_Order",
                    "Sales_Order_Line",
                    "Firm",
                    "Firm_Account",
                    "Firm_Contact",
                    "Product",
                    "Product_Class",
                    "Product_Vendor",
                    "Rate",
                    "Rate_Discount",
                    "Rate_Discount_Item",
                    "Rate_Discount_Period",
                    "Rate_Discount_Value",
                    "Rate_Item",
                    "Rate_Period",
                    "Rate_Value",
                ]

                schema_text = build_schema_text(
                    schema_df,
                    include_only_tables=TABLE_WHITELIST,
                    exclude_tables=["sysdiagrams"],
                )

                select_chain = SELECT_PROMPT | get_llm() | StrOutputParser()
                select_sql = select_chain.invoke({
                    "history": history_str,
                    "question": user_q,
                    "schema_text": schema_text
                }).strip()

                st.code(select_sql, language="sql")

                # Step 3: Execute SELECT
                df = run_select_sql(eng, select_sql)
                df = coerce_numeric_objects(df)

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
                column_config = build_column_config(df)

                st.dataframe(
                    df.head(50),
                    column_config=column_config,
                    use_container_width=True
                )


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
                st.info(" Analyzing existing results (no new SQL query)...")

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
                    st.dataframe(
                        display_df.head(20),
                        column_config=build_column_config(display_df),
                        use_container_width=True
                    )

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