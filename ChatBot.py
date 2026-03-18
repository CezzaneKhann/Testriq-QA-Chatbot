import streamlit as st
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

st.title("Testriq QA Assistant")
st.write("Powered by Groq + Llama 3")

# System prompt - gives chatbot its personality
system_prompt = {
    "role": "system",
    "content": """You are a helpful QA Assistant for Testriq QA Lab, 
    a professional software testing company. You help clients and team members 
    understand software testing concepts, explain test results, answer questions 
    about QA processes, and provide guidance on testing strategies including 
    manual testing, automation testing, performance testing, and AI/LLM testing. 
    Always be professional, clear, and helpful. If asked about LLM testing or 
    AI model evaluation, explain concepts in simple terms."""
}

# Store chat history
if "messages" not in st.session_state:
    st.session_state.messages = [system_prompt]

# Display previous messages (skip system prompt)
for message in st.session_state.messages:
    if message["role"] == "system":
        continue
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Take user input
user_input = st.chat_input("Type your message here...")

if user_input:
    # Show user message
    with st.chat_message("user"):
        st.write(user_input)

    # Save user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Get response from Groq
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=st.session_state.messages
    )
    reply = response.choices[0].message.content

    # Show bot response
    with st.chat_message("assistant"):
        st.write(reply)

    # Save bot response
    st.session_state.messages.append({"role": "assistant", "content": reply})