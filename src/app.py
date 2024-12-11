from langchain_google_genai import ChatGoogleGenerativeAI
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
import spacy
import asyncio
from spacy.matcher import PhraseMatcher
import re
import matplotlib.pyplot as plt
import pandas as pd
import io
import base64


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

nlp = spacy.load("en_core_web_lg")
# Define your database column names and associated synonyms
column_names = {
    "IndepYear": ["independent year", "independence year", "year of independence", "independent"],
    "Population": ["population", "number of people", "inhabitants"],
    "Continent": ["continent", "region"],
    # Add other columns and synonyms as needed
}

def preprocess_user_query(user_query, column_synonyms):
    # Replace synonyms with actual column names in the user query
    for column, synonyms in column_synonyms.items():
        for synonym in synonyms:
            pattern = re.compile(rf'\b{re.escape(synonym)}\b', re.IGNORECASE)
            user_query = pattern.sub(column, user_query)
    return user_query

# Create PhraseMatcher object and add synonym patterns to it
# Add patterns to the PhraseMatcher
matcher = PhraseMatcher(nlp.vocab, attr="LOWER")

for column, synonyms in column_names.items():
    patterns = [nlp.make_doc(text.lower()) for text in synonyms]  # Ensuring synonyms are in lowercase
    matcher.add(column, patterns)
    print(f"Added patterns for {column}: {patterns}")

def match_column(input_text):
    doc = nlp(input_text.lower().strip())  # Normalize input
    matches = matcher(doc)

    if matches:
        # Return the first matching column
        match_id, start, end = matches[0]
        return nlp.vocab.strings[match_id]
    
    # Fallback to similarity check if no exact matches found
    input_doc = nlp(input_text.lower())
    best_match = None
    best_similarity = 0.0
    for column, synonyms in column_names.items():
        for synonym in synonyms:
            synonym_doc = nlp(synonym.lower())
            similarity = input_doc.similarity(synonym_doc)
            if similarity > best_similarity and similarity > 0.8:  # Threshold for similarity
                best_match = column
                best_similarity = similarity

    return best_match

# Example Usage:
user_input = "I want to know about the YEAR OF INDEPENDENCE"
matched_column = match_column(user_input)
if matched_column:
    print(f"Matched user input to database column: {matched_column}")
else:
    print("No matching column found.")

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
    llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            stream=True,
            max_output_tokens=4096,
        )

    def get_schema(_):
        return db.get_table_info()
    
    return ( 
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm
        | StrOutputParser()
    )

def setup_event_loop():
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            asyncio.set_event_loop(asyncio.new_event_loop())
    except RuntimeError as e:
        asyncio.set_event_loop(asyncio.new_event_loop())

setup_event_loop()

# This function takes a user query, the database connection, and chat history to generate a response. 
# It invokes the SQL chain to generate a SQL query based on the user’s question and provides a clear response in natural language.
def get_response(user_query: str, db: SQLDatabase, chat_history: list):
    loop = asyncio.get_event_loop_policy().get_event_loop()
    if not loop.is_running():
        asyncio.set_event_loop(asyncio.new_event_loop())
    
    preprocessed_query = preprocess_user_query(user_query, column_names)
    print(f"Preprocessed Query: {preprocessed_query}")  # Debugging step

    # Match columns to confirm valid mapping
    matched_column = match_column(preprocessed_query)
    if matched_column:
        print(f"Matched user input to database column: {matched_column}")
    else:
        print("No matching column found.")

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
            - Do NOT reinterpret column names. Use them as they appear in the schema.
            - Use SELECT statements only. Avoid data modification commands.
            - Ensure the column names match exactly as provided in the schema (e.g., IndepYear, not Independence_Year).
            - If the user asks for a graph, provide a visual graph.
            
            Here is the sql database tables, please refer to this:
            CREATE TABLE `city` (
            `ID` int NOT NULL AUTO_INCREMENT,
            `Name` char(35) NOT NULL DEFAULT '',
            `CountryCode` char(3) NOT NULL DEFAULT '',
            `District` char(20) NOT NULL DEFAULT '',
            `Population` int NOT NULL DEFAULT '0',
            PRIMARY KEY (`ID`),
            KEY `CountryCode` (`CountryCode`),
            CONSTRAINT `city_ibfk_1` FOREIGN KEY (`CountryCode`) REFERENCES `country` (`Code`) )

            CREATE TABLE `country` (
            `Code` char(3) NOT NULL DEFAULT '',
            `Name` char(52) NOT NULL DEFAULT '',
            `Continent` enum('Asia','Europe','North America','Africa','Oceania','Antarctica','South America') NOT NULL DEFAULT 'Asia',
            `Region` char(26) NOT NULL DEFAULT '',
            `SurfaceArea` decimal(10,2) NOT NULL DEFAULT '0.00',
            `IndepYear` smallint DEFAULT NULL,
            `Population` int NOT NULL DEFAULT '0',
            `LifeExpectancy` decimal(3,1) DEFAULT NULL,
            `GNP` decimal(10,2) DEFAULT NULL,
            `GNPOld` decimal(10,2) DEFAULT NULL,
            `LocalName` char(45) NOT NULL DEFAULT '',
            `GovernmentForm` char(45) NOT NULL DEFAULT '',
            `HeadOfState` char(60) DEFAULT NULL,
            `Capital` int DEFAULT NULL,
            `Code2` char(2) NOT NULL DEFAULT '',
            PRIMARY KEY (`Code`)
            
            CREATE TABLE `countrylanguage` (
            `CountryCode` char(3) NOT NULL DEFAULT '',
            `Language` char(30) NOT NULL DEFAULT '',
            `IsOfficial` enum('T','F') NOT NULL DEFAULT 'F',
            `Percentage` decimal(4,1) NOT NULL DEFAULT '0.0',
            PRIMARY KEY (`CountryCode`,`Language`),
            KEY `CountryCode` (`CountryCode`),
            CONSTRAINT `countryLanguage_ibfk_1` FOREIGN KEY (`CountryCode`) REFERENCES `country` (`Code`)
        
        
        Conversation History: {chat_history}
        
        SQL Query: <SQL>{query}</SQL>
        
        User Question: {question}
        
        SQL Response: {response}
        
        Provide a natural language response to the user’s question based on the SQL response above:
        
    """

    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            stream=True,
            max_output_tokens=4096,
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
    print("User Query:", user_query)

    try:
        # Invoke the chain and get the query
        query_response = chain.invoke({
            "question": user_query,
            "chat_history": chat_history,
        })

        print(f"Query Response Content: {query_response}")

        # Parse the table data from the chatbot's response
        rows = []
        columns = []
        for i, line in enumerate(query_response.split("\n")):
            if i == 0:
                columns = [col.strip() for col in line.strip("|").split("|")]
            elif i > 1:  # Skip header separator
                row = [val.strip() for val in line.strip("|").split("|")]
                rows.append(row)

        # If the user requested a graph
        if "graph" in user_query.lower() or "chart" in user_query.lower():
            data = pd.DataFrame(rows, columns=columns)
            chart_type = "bar"  # Default chart type; can be inferred or customized
            return generate_graph_from_dataframe(data, chart_type)

        # Return the original chatbot response if no graph is requested
        return query_response

    except Exception as e:
        return f"Error: {e}"

def generate_graph_from_dataframe(df: pd.DataFrame, chart_type="bar"):
    """
    Generate a graph from a DataFrame.
    """
    # Convert numeric columns to numeric types
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Create the plot
    plt.figure(figsize=(10, 6))
    if chart_type == "bar":
        df.plot(kind="bar", x=df.columns[0], y=df.columns[1:], ax=plt.gca())
    elif chart_type == "line":
        df.plot(kind="line", x=df.columns[0], y=df.columns[1:], ax=plt.gca())
    elif chart_type == "scatter" and len(df.columns) >= 3:
        plt.scatter(df[df.columns[1]], df[df.columns[2]])
        plt.xlabel(df.columns[1])
        plt.ylabel(df.columns[2])
    else:
        raise ValueError("Unsupported chart type or insufficient columns.")

    plt.title("Generated Graph")
    plt.xlabel(df.columns[0])
    plt.ylabel("Values")
    plt.tight_layout()

    # Save the plot to a BytesIO stream
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    plt.close()

    # Encode the image to base64 to display in Streamlit
    img_base64 = base64.b64encode(buffer.read()).decode()
    return f"![Graph](data:image/png;base64,{img_base64})"

# Loads sensitive environment variables like API keys from a .env file so they can be used securely within the code.
load_dotenv()

test_query = "Which countries gained independence after 1950 and have a population greater than 50 million?"
print(preprocess_user_query(test_query, column_names))


st.image("./VisfutureLogo.png")

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