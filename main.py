# main.py
import os
from dotenv import load_dotenv
import streamlit as st
from streamlit_chat import message
import faiss
from langchain import OpenAI
from langchain.chains import VectorDBQAWithSourcesChain
import pickle

# Load environment variables
load_dotenv()

# Set the OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Load the LangChain.
index = faiss.read_index("docs.index")

with open("faiss_store.pkl", "rb") as f:
    store = pickle.load(f)

store.index = index
chain = VectorDBQAWithSourcesChain.from_llm(llm=OpenAI(
    api_key=OPENAI_API_KEY), vectorstore=store)

# From here down is all the StreamLit UI.
st.set_page_config(page_title="Blendle Notion QA Bot", page_icon=":robot:")
st.header("Blendle Notion QA Bot")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


def get_text():
    input_text = st.text_input("You: ", "Hello, how are you?", key="input")
    return input_text


user_input = get_text()

if user_input:
    print(f"User input: {user_input}")  # Add this line for debugging
    result = chain({"question": user_input})
    output = f"Answer: {result['answer']}\nSources: {result['sources']}"

    st.write(output)
