# streamlit_app.py

import streamlit as st
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
import dotenv
import os
from InstructorEmbedding import INSTRUCTOR
from langchain.embeddings import HuggingFaceInstructEmbeddings
import google.generativeai as genai

# Load environment variables
dotenv.load_dotenv()

# Configure Google Generative AI
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

llm = genai.GenerativeModel('gemini-pro')

# Streamlit UI
st.title("AI Response Generator")

# Input text box
input_text = st.text_area("Enter your input text:", "What's the point of making myself less accessible?")

loader = DirectoryLoader('./', glob="./*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-base", model_kwargs={"device": "cpu"})

persist_directory = 'db'
embedding = instructor_embeddings

vectordb = Chroma.from_documents(documents=texts, embedding=embedding, persist_directory=persist_directory)
vectordb.persist()
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

# Function to generate response
def generate_response(input_text):

    docs = retriever.get_relevant_documents(input_text)

    text = ""
    for doc in docs:
        text += doc.page_content

    new_input_text = f"Given the below details:\n{text}\n\n do the following \n{input_text}\n"
    response = llm.generate_content(new_input_text)

    return response.text



# Button to generate response
if st.button("Generate Response"):
    # Generate response
    response_text = generate_response(input_text)

    # Display the response
    st.subheader("Generated Response:")
    st.write(response_text)
