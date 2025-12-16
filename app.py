import os
import sys
import types
import asyncio

#Patch asyncio to prevent runtime loop errors
try:
    loop = asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

#Patch Streamlit's local_sources_watcher to ignore torch
import streamlit.watcher.local_sources_watcher as watcher

original_get_module_paths = watcher.get_module_paths

def patched_get_module_paths(module: types.ModuleType):
    if module.__name__.startswith("torch"):
        return []
    return original_get_module_paths(module)

watcher.get_module_paths = patched_get_module_paths
os.environ["STREAMLIT_SERVER_RUN_ON_SAVE"] = "false"


try:
    loop = asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
from langchain.prompts import PromptTemplate  # Import basic PromptTemplate (unused in this snippet, but available if needed). # Import ChatPromptTemplate to build a chat-style prompt.
from langchain_core.output_parsers import StrOutputParser  # Import a string output parser to process the LLM response.
from langchain_community.llms import Ollama  # Import the Ollama LLM wrapper from the LangChain Community package.
from glob import glob
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, trim_messages 
import streamlit as st  
import base64
from langchain.chains import RetrievalQA
from langchain_core.runnables import RunnableMap, RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from pathlib import Path                                                         
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings  # ✅ updated
from langchain_community.vectorstores import Chroma, FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from PIL import Image
from langchain_core.chat_history import InMemoryChatMessageHistory
from streamlit.components.v1 import html






from dotenv import load_dotenv
load_dotenv() 

os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY") # Load the environment variables from the .env file.
os.environ['OpenAI_API_KEY'] = os.getenv("OpenAI_API_KEY")
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
embeddings = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")# Load the environment variables from the .env file.
groq_api_key = os.getenv("GROQ_API_KEY") # Get the API key for Groq from the environment variables.

from langchain_core.messages import HumanMessage, AIMessage

def summarize_conversation(llm, messages):
    history_text = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
        for m in messages
    )

    prompt = (
        "Here is the conversation between a user and an AI assistant:\n\n"
        f"{history_text}\n\n"
        "Please provide a detailed summary of the conversation in bullet points."
    )

    return llm.invoke(prompt)



# Load and display logo
# Display centered logo using custom HTML
# Load and display centered logo
logo = Image.open("logo (2).png")
st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
st.image(logo, width=200)
st.markdown("</div>", unsafe_allow_html=True)

# App title and intro
st.title('Welcome to CamAI Chatbot')
st.subheader('💬 Chat with your documents')
st.write('📄 Upload your documents and ask questions about them.')
st.write('🔍 You can also upload multiple documents at once.')
# 🔶 Apply Opensky-styled button colors
st.markdown("""
    <style>
        div.stButton > button:first-child {
            background-color: #f58220;
            color: white;
            font-weight: bold;
            border: none;
            border-radius: 5px;
            padding: 0.6rem 1.2rem;
        }
        div.stButton > button:first-child:hover {
            background-color: #e6731a;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)




# ✅ Use Groq API key from environment (already loaded into groq_api_key)
if not groq_api_key:
    st.error("GROQ_API_KEY not found. Please set it in your .env or environment variables.")
    st.stop()

# ✅ Initialize LLM once, without any user password input
llm = ChatGroq(api_key=groq_api_key, model="llama-3.3-70b-versatile", temperature=0.7)

session_id = st.text_input("Session ID", value="session_id")

if 'store' not in st.session_state:
    st.session_state.store = {}

uploaded_files = st.file_uploader("Upload your documents", type=["txt", "pdf"], accept_multiple_files=True)

## Process the file 
if uploaded_files:
    documents = []
    for upladed_file in uploaded_files:
        tempdf = f"./temp.pdf"
        with open(tempdf, "wb") as file:
            file.write(upladed_file.getvalue())
            file_name = upladed_file.name

        loader = PyPDFLoader(tempdf)
        docs = loader.load()
        documents.extend(docs)

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=256)
    chunks = text_splitter.split_documents(documents)
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriver = vector_store.as_retriever()

    contextualize_q_system_prompt = (
        "You are a helpful, precise, and context-aware assistant. "
        "Use only the information provided in the retrieved documents and the conversation history to answer user questions. "
        "If the question is not answerable based on the given context, respond with: "
        "'I'm not sure about that based on the available information.' "
        "Do not assume or hallucinate any facts. Maintain a clear and conversational tone."
    )

    contextualize_q_system_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    ## Answer Question
    system_prompt = (
        "You are a helpful, precise, and context-aware assistant. "
        "Use only the information provided in the retrieved documents to answer user questions. "
        "If the answer is not present or cannot be confidently determined from the documents, respond with: "
        "'I'm not sure about that based on the available information.' "
        "Do not make assumptions or generate information that is not grounded in the context. "
        "Always provide clear and concise responses in a natural, conversational tone.\n\n"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(retriver, question_answer_chain)

    def get_session_history(session: str) -> BaseChatMessageHistory:
        if session not in st.session_state.store:
            st.session_state.store[session] = InMemoryChatMessageHistory()
        return st.session_state.store[session]

    conversational_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_message_key="input",
        history_messages_key="chat_history",
        output_message_key="answer"
    )

    user_input = st.text_input("What question do you want to ask?")
    if user_input:
        session_history = get_session_history(session_id)
        response = conversational_chain.invoke(
            {"input": user_input},
            config={
                "configurable": {
                    "session_id": session_id
                }
            }
        )

        # ✅ Manually log the interaction to the history
        session_history.add_user_message(user_input)
        session_history.add_ai_message(response['answer'])

        st.success(f"Assistant: {response['answer']}")
        with st.expander("🗂️ Chat History"):
            for msg in session_history.messages:
                role = "🧠 Assistant" if isinstance(msg, AIMessage) else "🧑 You"
                st.markdown(f"**{role}:** {msg.content}")

        # 🧠 Chat Summary Button
        if st.button("🧾 Summarize This Chat"):
            with st.spinner("Summarizing..."):
                summary_response = summarize_conversation(llm, session_history.messages)
                st.markdown("### 📝 Summary of the Chat")
                st.info(summary_response)
