from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import streamlit as st
from langchain_groq import ChatGroq
import os

# Here, we are importing modules from various libraries. These libraries help us load environment variables, 
# handle messages, create chat templates, connect to a SQL database, parse outputs, and build a web application.

# Checks if there is an existing chat history. If not, it creates a new one with an AI message saying hello. 
# This ensures our chat keeps track of previous messages.
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content ="Hello! I am your SQL ChatBot. Ask me anything about your SQL database.")
    ]

# This function takes database connection details (user, password, host, port, database name) and uses them to create a connection string. 
# This connection string connects the chatbot to a SQL database.
def init_database(user: str, password: str, host: str, port: str, database: str) -> SQLDatabase:
    db_uri = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"
    return SQLDatabase.from_uri(db_uri)

def validate_columns(query: str, schema: dict):
    # Extract all column names from the schema
    valid_columns = {col for table in schema.values() for col in table["columns"]}
    
    # Check if all columns in the query exist in the schema
    for word in query.split():
        if word in valid_columns:
            continue
        elif "(" in word:  # Ignore derived columns like GNP/Population
            continue
        else:
            raise ValueError(f"Invalid column in query: {word}")

def get_sql_chain(db):
    template = """
        You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
        Based on the table schema below, write a SQL query that would answer the user's question. Take the conversation history into account.
        
        <Schema>

        **Important Instructions:**
        - Do NOT perform any data modification commands such as DELETE, UPDATE, or INSERT. Use only SELECT statements.
        - If the user asks to "remove" or "exclude" specific entries, interpret this as a request to filter results. For example:
            - User says: "Remove Algeria from the table."
            - SQL Query: SELECT * FROM country WHERE Name != 'Algeria';
        - Ensure you distinguish carefully between similar terms in the schema, like `Code` vs. `Name`, to avoid incorrect interpretations.
        - Avoid any nested queries or unnecessary complexity unless specifically requested by the user.
        - **Present results in table format with columns and rows using `|` for separators.** Avoid numbered or bullet lists for structured data.
        - If the user asks for rows to be removed or excluded, describe the filtered output, ensuring no database modifications occur unless explicitly requested.
        - Use JOIN statements where necessary to combine data from multiple tables based on relationships.
        - Avoid Cartesian products by always including proper JOIN conditions.
        - Use column names exactly as they appear in the database schema. For example, `LifeExpectancy`, not `Life Expectancy`.
        - Ensure all column names and tables referenced in the query exist in the schema.
        - Do NOT use column names that do not exist in the schema.
        - If the user asks for a derived value (e.g., GNP per capita), calculate it explicitly using existing columns. For example:
            - User: Show the top countries by GNP per capita.
            - SQL Query: SELECT Name, (GNP / Population) AS GNPperCapita FROM country WHERE Population > 0 ORDER BY GNPperCapita DESC LIMIT 10;
        - Always validate column names against the schema provided.
        - If no matching column exists, inform the user that the column is not available.
        - Write only the SQL query and nothing else.

        
        
        
        Conversation History: {chat_history}
        
        Write only the SQL query and nothing else. Do not wrap the SQL query in any other text, not even backticks. Ensure you consider the difference between column names, such as `Code` versus `Name`, to avoid misinterpretation.
    
        For example:
        Question: How many countries start with the letter Z?
        SQL Query: SELECT COUNT(*) FROM country WHERE Name LIKE 'Z%';
        
        For example:
        Question: which 3 countries have the largest populations?
        SQL Query: SELECT Name FROM country ORDER BY Population DESC LIMIT 3;
        Question: List all countries in Asia
        SQL Query: SELECT Name FROM country WHERE Continent = 'Asia';
        
        Your turn:
        
        Question: {question}
        SQL Query:
        
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatGroq(
    model="mixtral-8x7b-32768",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    streaming = True,
    # other params...
)

    def get_schema(_):
        return db.get_table_info()
    
    return ( 
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm
        | StrOutputParser()
    )



# This function takes a user query, the database connection, and chat history to generate a response. 
# It invokes the SQL chain to generate a SQL query based on the user’s question and provides a clear response in natural language.
def get_response(user_query: str, db: SQLDatabase, chat_history: list):
    sql_chain = get_sql_chain(db)

    template = """
        You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
        Based on the table schema below, question, SQL query, and SQL response, write a natural language response that answers the user’s question in simple and clear terms.
        
        **Important Instructions:**
            - If the user requests the "removal" or "exclusion" of certain rows, clarify in your response that these rows are excluded based on a filter in the SQL query rather than permanently deleted.
            - Use concise and straightforward language in the response to ensure clarity.
            - If the response involves an aggregated list, summarize results succinctly.
            - **Present results in table format with columns and rows using `|` for separators.** Avoid numbered or bullet lists for structured data.
            - If the user asks for rows to be removed or excluded, describe the filtered output, ensuring no database modifications occur unless explicitly requested.
            - Use JOIN statements where necessary to combine data from multiple tables based on relationships.
            - Avoid Cartesian products by always including proper JOIN conditions.
            - Use column names exactly as they appear in the database schema. For example, `LifeExpectancy`, not `Life Expectancy`.
            
            

        
        
        Conversation History: {chat_history}
        
        SQL Query: <SQL>{query}</SQL>
        
        User Question: {question}
        
        SQL Response: {response}
        
        Provide a natural language response to the user’s question based on the SQL response above:
        
    """

    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatGroq(
    model="mixtral-8x7b-32768",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    streaming = True,
    # other params...
)
    
    chain = (
        RunnablePassthrough.assign(query=sql_chain).assign(
            schema=lambda _: db.get_table_info(),
            response=lambda vars: db.run(vars["query"]),
        )
        | prompt
        | llm
        | StrOutputParser()
    )


    return chain.invoke({
        "question": user_query,
        "chat_history": chat_history,
    })

# Loads sensitive environment variables like API keys from a .env file so they can be used securely within the code.
load_dotenv()

st.image("./ChatBot/VisfutureLogo.png")

st.title("ChatBot")

st.sidebar.subheader("Settings")
st.sidebar.write("This is a simple ChatBot application using MySQL. Connect to the database and start using ChatBot.")

# Sidebar inputs
host = st.sidebar.text_input("Host", value="localhost", key="Host")
port = st.sidebar.text_input("Port", value="3306", key="Port")
user = st.sidebar.text_input("User", value="root", key="User")
password = st.sidebar.text_input("Password", type="password", value="", key="Password")
database = st.sidebar.text_input("Database", value="world", key="Database")

# When the “Connect” button is pressed, it tries to connect to the database with the entered details. 
# If successful, it shows a success message; otherwise, it displays an error message.
if st.sidebar.button("Connect"):
    with st.spinner("Connecting to database..."):
        try:
            db = init_database(user, password, host, port, database)
            st.session_state.db = db
            st.sidebar.success("Connected to database!")
        except Exception as e:
            st.error(f"Error: {e}")

# Display chat messages on the main page
st.subheader("Chat")

for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)

# hows the chat history on the main page, differentiating between AI and human messages.
user_query = st.chat_input("Type a message...")
if user_query is not None and user_query.strip() != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        response = get_response(user_query, st.session_state.db, st.session_state.chat_history)
        st.markdown(response)

    st.session_state.chat_history.append(AIMessage(content=response))