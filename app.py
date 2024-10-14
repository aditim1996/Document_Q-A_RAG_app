import os
import streamlit  as st
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain ## to get the realevent docs
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS ## vectorstore database 
from langchain_community.document_loaders import PyPDFDirectoryLoader ## reading some files from folder
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings #vector embedding techniques 

from dotenv import load_dotenv


load_dotenv()

## loading the groq and google api ##

groq_api_key = os.getenv("GROQ_API_KEY")
os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")

st.title("Gemma Model Document Q&A")

llm = ChatGroq(groq_api_key=groq_api_key,model_name="gemma-7b-it")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most acurate response based on the questio
    <context>
    {context}
    Question:{input}


    """
)

def vector_embedding():#### reading all the doc form pdf convert into chunks and apply embedding & storing in vector db( faiss)
    if "vectors" not in st.session_state:
        st.session_state.embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state.loader=PyPDFDirectoryLoader("./us_census")## Data ingestion
        st.session_state.docs=st.session_state.loader.load()## loading the documents
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap = 200)
        st.session_state.final_documents= st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)
        
prompt1 = st.text_input("What you want to ask from the documents?")

if st.button("Documents Embeddings"):
    vector_embedding()
    st.write("Vector Store DB IS Ready")

import time

if prompt1:
    document_chain = create_stuff_documents_chain(llm,prompt)       
    retriever = st.session_state.vectors.as_retriever() ## will be able to retrive the particular info a/c to input
    retrieval_chain=create_retrieval_chain(retriever,document_chain)
    
    
    start = time.process_time()
    response = retrieval_chain.invoke({'input':prompt1})
    print("Response time :",time.process_time()-start)
    st.write(response['answer'])    
    
    with st.expander("Documnent Similarity Search"):
        #Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("-------------------------")
            







