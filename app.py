# app.py
import os
import time
import pandas as pd
from flask import Flask, request, jsonify
from threading import Thread
import requests
import streamlit as st
from groq import Groq, GroqError
import matplotlib.pyplot as plt

# ===============================
# CONFIGURATION
# ===============================
DATA_PATH = "financial_analysis_results.csv"  # or your Excel path
GROQ_API_KEY = "API_KEY"  # replace with your Groq API key

# ===============================
# LOAD FINANCIAL DATA
# ===============================
def load_financial_data(path):
    if path.endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)

    numeric_cols = [
        "Total Revenue", "Net Income", "Total Assets",
        "Total Liabilities", "Cash Flow from Operating Activities"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Calculate growth metrics
    df['Revenue Growth (%)'] = df.groupby('Company')['Total Revenue'].pct_change() * 100
    df['Net Income Growth (%)'] = df.groupby('Company')['Net Income'].pct_change() * 100
    return df

df = load_financial_data(DATA_PATH)

# ===============================
# RULE-BASED RESPONSE
# ===============================
def rule_based_response(user_text):
    text = user_text.lower()
    companies = ['microsoft', 'apple', 'tesla']
    company = next((c for c in companies if c in text), None)

    if "total revenue" in text and company:
        total = df[df['Company'].str.lower() == company]['Total Revenue'].sum()
        return f"{company.capitalize()}'s total revenue (all years) is ${total:,.0f}."

    if "net income" in text and any(w in text for w in ["change", "trend"]):
        comp_data = df[df['Company'].str.lower() == company].sort_values('Fiscal Year')
        if comp_data.shape[0] < 2:
            return f"Not enough data for {company.capitalize()} to compute net income trend."
        prev, last = comp_data.iloc[-2]['Net Income'], comp_data.iloc[-1]['Net Income']
        change = last - prev
        pct = (change / prev) * 100 if prev != 0 else float('inf')
        direction = "increased" if change > 0 else "decreased"
        return (f"{company.capitalize()}'s net income {direction} by ${abs(change):,.0f} "
                f"({pct:.1f}%) from {int(comp_data.iloc[-2]['Fiscal Year'])} "
                f"to {int(comp_data.iloc[-1]['Fiscal Year'])}.")

    if "highest revenue" in text or "top revenue" in text:
        grp = df.groupby("Company", as_index=False)["Total Revenue"].sum()
        top = grp.sort_values("Total Revenue", ascending=False).iloc[0]
        return f"{top['Company']} has the highest total revenue: ${top['Total Revenue']:,.0f}."

    return None

# ===============================
# GENERATE COMPACT SUMMARY FOR LLM
# ===============================
def generate_summary_context(df, company=None, max_years=5):
    if company:
        dfc = df[df['Company'].str.lower() == company.lower()]
        if dfc.empty:
            return f"No data found for {company}."
    else:
        dfc = df.copy()

    # Keep last max_years
    dfc = dfc.sort_values('Fiscal Year').tail(max_years)
    key_cols = ['Fiscal Year', 'Total Revenue', 'Net Income', 'Revenue Growth (%)',
                'Net Income Growth (%)', 'Total Assets', 'Total Liabilities',
                'Cash Flow from Operating Activities']
    dfc = dfc[[c for c in key_cols if c in dfc.columns]]
    return dfc.to_markdown(index=False)

# ===============================
# GROQ LLM CHATBOT
# ===============================
class GroqFinancialChatbot:
    def __init__(self, df):
        self.df = df
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model = "meta-llama/llama-4-scout-17b-16e-instruct"

    def chat(self, user_text):
        companies = ["microsoft", "apple", "tesla"]
        company = next((c for c in companies if c in user_text.lower()), None)
        context = generate_summary_context(self.df, company=company, max_years=5)

        system_prompt = f"""
        You are a financial assistant AI. Here is the company financial data:

        {context}

        Answer clearly and accurately, using the financial data for your analysis report.
        If no data is available, answer generally.
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text}
                ],
                temperature=0.5,
                max_tokens=300
            )
            return response.choices[0].message.content
        except GroqError as e:
            return f" LLM Error: {e}"

BOT = GroqFinancialChatbot(df=df)

# ===============================
# FLASK BACKEND
# ===============================
app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    data = request.json or {}
    user_text = data.get("text", "").strip()
    if not user_text:
        return jsonify({'error': 'text is required'}), 400

    rule_answer = rule_based_response(user_text)
    if rule_answer:
        return jsonify({"response": rule_answer})

    reply = BOT.chat(user_text)
    return jsonify({"response": reply})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

def run_flask():
    app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)

# ===============================
# STREAMLIT FRONTEND
# ===============================
def run_streamlit():
    st.set_page_config(page_title="GFC Financial Chatbot", layout="centered")
    st.title("🤖 GFC Financial Insight Chatbot")
    st.markdown("Ask about revenue, profit trends, or financial insights for Microsoft, Tesla, and Apple.")

    FLASK_URL = "http://127.0.0.1:5000/chat"

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Show past messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input
    user_input = st.chat_input("Ask your financial question...")

    if user_input:
        st.chat_message("user").markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Rule-based + Groq
        try:
            res = requests.post(FLASK_URL, json={"text": user_input})
            if res.status_code == 200:
                bot_reply = res.json().get("response", "No reply from Groq LLM.")
            else:
                bot_reply = f" Error: {res.text}"
        except Exception as e:
            bot_reply = f" Cannot reach Flask server: {e}"

        # Display response
        st.chat_message("assistant").markdown(bot_reply)
        st.session_state.messages.append({"role": "assistant", "content": bot_reply})

# ===============================
# MAIN
# ===============================
if __name__ == "__main__":
    flask_thread = Thread(target=run_flask)
    flask_thread.daemon = True
    flask_thread.start()
    time.sleep(2)

    run_streamlit()
