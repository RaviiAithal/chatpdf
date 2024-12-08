import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key = os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_doc):
    text = ""
    for pdf in pdf_doc:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 10000, chunk_overlap = 1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectors_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding = embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the questions detailed as possible from the provided context, in case of no answer from the context tell no answer
    Context :\n {context}?\n
    Question:\n {question}\n  
    
    ANS:
    """

    model = ChatGoogleGenerativeAI(model = "gemini-pro", temperature = 0.3)
    prompt = PromptTemplate(template = prompt_template, input_variables = ["context","question"])
    chain = load_qa_chain(model, chain_type = "stuff", prompt = prompt)
    return chain

# def user_input(user_question):
#     embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
#
#     new_db = FAISS.load_local("faiss_index, embeddings")
#     docs = new_db.similarity_search(user_question)
#     chain = get_conversational_chain()
#
#     response = chain(
#         {"input_documents":docs, "question":user_question}
#         , return_only_outputs = True
#     )
#
#     print(response)
#     st.write("Reply: ", response["output_text"])
#

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)  # Enable deserialization
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

def main():
    st.set_page_config(page_title="PDF Q&A with AI", layout="wide")
    st.title("📄 PDF Q&A with Google Generative AI")
    st.markdown("Upload PDFs and ask AI questions based on their content.")

    # Sidebar for PDF upload
    st.sidebar.header("Upload PDFs")
    uploaded_files = st.sidebar.file_uploader(
        "Select one or more PDF files", type=["pdf"], accept_multiple_files=True
    )

    if uploaded_files:
        st.sidebar.success("Files uploaded successfully!")
        with st.spinner("Processing PDFs..."):
            pdf_text = get_pdf_text(uploaded_files)
            text_chunks = get_text_chunks(pdf_text)
            get_vectors_store(text_chunks)
        st.sidebar.success("PDFs processed and indexed!")

    # Main area for user input and responses
    st.markdown("### Ask Your Questions")
    user_question = st.text_input("Enter your question below:")
    if user_question and uploaded_files:
        with st.spinner("Generating response..."):
            reply = user_input(user_question)
        st.success("Response:")
        st.write(reply)
    elif not uploaded_files:
        st.warning("Please upload PDF(s) first.")


if __name__ == "__main__":
    main()