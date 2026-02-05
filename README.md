# Sales Analysis ChatBot (SQL + LangChain + Streamlit)

This project is an AI-powered **Sales Analysis ChatBot** built with **Streamlit**, **LangChain**, and **Groq models**.  
It connects to your **SQL Server database** and allows you to ask natural-language questions that it automatically converts into SQL queries and executes.

---

## STEP-BY-STEP SETUP GUIDE

### 1. Clone the Repo and run on the correct branch

1. Open PowerShell and go to where you want the project folder :

cd "folderpath"
git clone https://github.com/giethang/ChatBot.git
cd ChatBot

2. Switch to the correct Git branch

This project must be run from the NovusIndividualDatabase_C branch.

3. Run:

git fetch
git checkout NovusIndividualDatabase_C

4. Verify you’re on the correct branch:

git branch

You should see:

- NovusIndividualDatabase_C

# 2) Create a virtual env with the launcher

py -3.12 -m venv .venv

# 3) _IMPORTANT_: Activate the venv

.venv\Scripts\activate

After activation, your PowerShell prompt should show:

(.venv) PS C:\Users\<yourname>\Desktop\Novus\Novus Chatbot\ChatBot>

PS. Make sure you are on python version 3.12
To check your python version, run:

py --version

# 4) Upgrade pip

py -m pip install --upgrade pip

# 5) Install deps

py -m pip install -r requirements.txt

# 6) Get the spaCy model your app.py uses

py -m spacy download en_core_web_sm

# 7) Create a .env file and add your API key

Right click on the project folder and create a new file called .env
Add the following lines to the file:

# === LangChain ===

LANGCHAIN_API_KEY=
LANGCHAIN_TRACING_V2=true

# === SQL Server Config ===

MSSQL_HOST=10.1.1.4
MSSQLUAT_HOST=172.20.30.10
MSSQL_PORT=1433
MSSQL_USER=
MSSQL_PASSWORD=
MSSQL_ODBC_DRIVER=ODBC Driver 17 for SQL Server

# === Groq (your app uses ChatGroq) ===

GROQ_API_KEY=

# 8) Run the app (if it’s Streamlit)

streamlit run src\app.py

# FAQ or common errors:

“Data source name not found / ODBC Driver 17…”

Install Microsoft ODBC Driver 17 for SQL Server (and restart terminal after install).

“Module not found”

You likely didn’t activate the venv or installed packages outside it.
Re-run:

.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
