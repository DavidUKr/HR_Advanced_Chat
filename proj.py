import streamlit as st
import os
import chromadb
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.callbacks import get_openai_callback
import PyPDF2
import pdfplumber
import pytesseract
import re
from pdfminer.high_level import extract_text
import warnings
from pdf2image import convert_from_path
import openai
openai.api_key=""
class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata if metadata is not None else {}

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        # Ignore the DeprecationWarning raised by PyPDF2
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=DeprecationWarning)
            reader = PyPDF2.PdfReader(file)
            text = ''
            for page_num in range(len(reader.pages)):
                page =  reader.pages[page_num]
                text += page.extract_text()
    return text

# Extract tables from PDF using pdfplumber
def extract_tables_from_pdf(pdf_path):
    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_tables = page.extract_tables()
            tables.extend(page_tables)
    return tables

# Extract images and text
def extract_images_and_text_from_pdf(pdf_path):
    images = []
    texts = []
    
    # Convert PDF pages to images
    pages = convert_from_path(pdf_path)
    
    # Extract text from images using pytesseract
    for page in pages:
        images.append(page)
        texts.append(pytesseract.image_to_string(page))
    
    return images, texts

# Load PDF using PyPDFLoader
pdf_path =  r"C:/Users/Talent2/Desktop/ness/employee_details.pdf"
text_from_pdf = extract_text_from_pdf(pdf_path)
tables_from_pdf = extract_tables_from_pdf(pdf_path)
images_from_pdf, image_texts_from_pdf = extract_images_and_text_from_pdf(pdf_path)

# Preprocess extracted data
cleaned_text = text_from_pdf
cleaned_tables = tables_from_pdf
cleaned_image_texts = image_texts_from_pdf

# Create Document objects with metadata
# Create Document objects with metadata
documents = [
    Document(page_content=cleaned_text, metadata={"type": "text"}),
    *[
        Document(page_content=table, metadata={"type": "table"})
        for table in cleaned_tables
    ],
    *[
        Document(page_content=image_text, metadata={"type": "image_text"})
        for image_text in cleaned_image_texts
    ]
]

print(documents)

# ChromaDB setup
embedding = OpenAIEmbeddings(api_key='')
text_contents = [doc.page_content for doc in documents]

text_contents = [str(text) for text in text_contents]

# Concatenate all text contents into a single string
all_text = "\n".join(text_contents)

# Pass the concatenated text to Chroma.from_texts()
vectordb = Chroma.from_texts(texts=all_text, embedding=embedding)

# RAG setup
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
qa = RetrievalQA.from_chain_type(
    llm, retriever=vectordb.as_retriever()
)

# Streamlit setup
st.title("Human Resources Advanced Companion")

def run_with_chromadb(chain, query):
    response = chain.invoke(query)
    return response['result']

# Question input
user_input = st.text_input("Ask your question here")

if user_input:
    output = run_with_chromadb(qa, user_input)
    st.write(output)
