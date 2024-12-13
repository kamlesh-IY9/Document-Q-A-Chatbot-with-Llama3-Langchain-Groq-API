import streamlit as st
import os
from langchain_groq.chat_models import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time

from dotenv import load_dotenv
load_dotenv()

## load the Groq API key
groq_api_key=os.environ['GROQ_API_KEY']
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

if "vector" not in st.session_state:
    st.session_state.embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    st.session_state.loader=WebBaseLoader("https://docs.smith.langchain.com/")
    st.session_state.docs=st.session_state.loader.load()
    
    st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
    st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)
    
st.title("ChatGroq Demo 💬")
llm = ChatGroq(groq_api_key=groq_api_key, model="llama-3.1-70b-versatile")

groq_api_key = os.getenv('GROQ_API_KEY')
if not groq_api_key:
    st.error("🚫 API Key is not set properly. Please check! 🔑")
else:
    st.markdown("<h3 style='color: green;'>✅ API Key loaded successfully! 🎉</h3>", unsafe_allow_html=True)
    # st.write("API Key loaded successfully.")

prompt=ChatPromptTemplate.from_template(
""" 
    Answer the questions based on the provided context only.
    please provide the most accurate response based on the question
    <context>
    {context}
    </context>
    Questions:{input}
"""
)

document_chain=create_stuff_documents_chain(llm,prompt)
retriever=st.session_state.vectors.as_retriever()
retrieval_chain=create_retrieval_chain(retriever,document_chain)

prompt1=st.text_input("Input your prompt here")

if prompt1:
    start=time.process_time()
    response=retrieval_chain.invoke({"input":prompt1})
    print("Response time : ",time.process_time() - start)
    print("response time - start - " ,start , "\nresponse time from process_time : ",time.process_time())
    st.write(response['answer'])

    # With a streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write("-----------------------------------------------")
    