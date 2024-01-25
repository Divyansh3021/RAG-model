from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
import dotenv
import os
from InstructorEmbedding import INSTRUCTOR
from langchain.embeddings import HuggingFaceInstructEmbeddings

loader = DirectoryLoader('./', glob="./*.pdf", loader_cls=PyPDFLoader)

documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

from langchain.embeddings import HuggingFaceInstructEmbeddings

instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl", 
                                                      model_kwargs={"device": "cpu"})


persist_directory = 'db'

embedding = instructor_embeddings

vectordb = Chroma.from_documents(documents=texts, 
                                 embedding=embedding,
                                 persist_directory=persist_directory)

vectordb.persist()
vectordb = None

vectordb = Chroma(persist_directory=persist_directory, 
                  embedding_function=embedding)

retriever = vectordb.as_retriever(search_kwargs={"k": 3})

input_text = "What's the point of making myself less accessible?"

docs = retriever.get_relevant_documents(input_text)

text = ""
for doc in docs:
    text += doc.page_content

import google.generativeai as genai

dotenv.load_dotenv()

genai.configure(api_key= os.getenv('GOOGLE_API_KEY'))

llm = genai.GenerativeModel('gemini-pro')

new_input_text = f"Given the below details:\n{text}\n\n do the following \n{input_text}\n"
response = llm.generate_content(new_input_text)
print(response.text)
