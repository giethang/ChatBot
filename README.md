# Sales Analysis ChatBot (SQL + LangChain + Streamlit)

This project is an AI-powered **Sales Analysis ChatBot** built with **Streamlit**, **LangChain**, and **Groq models**.  
It connects to your **SQL Server database** and allows you to ask natural-language questions that it automatically converts into SQL queries and executes.

---

## STEP-BY-STEP SETUP GUIDE

### 1. Unzip the Project

1. Locate and unzip the project folder (for example `Novus Chatbot.zip`).
2. Move the extracted folder to your desired location (e.g. Desktop).
3. Open **PowerShell** or **Command Prompt** and navigate to the project folder:
   ```bash
   cd "C:\Users\<yourname>\Desktop\Novus\Novus Chatbot\ChatBot"
   ```

# 2) Create a virtual env with the launcher

py -3.12 -m venv .venv

# 3) Activate the venv

.venv\Scripts\activate

After activation, your PowerShell prompt should show:

(.venv) PS C:\Users\<yourname>\Desktop\Novus\Novus Chatbot\ChatBot>

PS. Make sure you are on python version 3.12
To check your python version, run:

py --version

# 4) Upgrade pip

py -m pip install --upgrade pip

# ) Install deps

py -m pip install -r requirements.txt

# 7) Get the spaCy model your app.py uses

# 9) Create a .env file and add your API key

# === API KEYS ===

# === SQL SERVER CONFIGURATION ===

# 8) Run the app (if it’s Streamlit)

streamlit run src\app.py
