import multiprocessing as mp
from PyPDF2 import PdfReader
import pdfplumber
from pdf2image import convert_from_path
import pytesseract
import os
import warnings

# Define the extract_text_from_page function
def extract_text_from_page(page_content):
    return page_content.extract_text()

# Extract text from PDF using multiprocessing
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PdfReader(file)
        page_texts = [page.extract_text() for page in reader.pages]
        
    with mp.Pool(mp.cpu_count()) as pool:
        text = pool.map(str, page_texts)
    
    return ''.join(text)

# Extract tables from PDF using multiprocessing
def extract_tables_from_page(page):
    return page.extract_tables()

def extract_tables_from_pdf(pdf_path):
    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        page_tables = [page.extract_tables() for page in pdf.pages]
        
    with mp.Pool(mp.cpu_count()) as pool:
        tables = pool.map(lambda x: x, page_tables)
    
    return [table for sublist in tables for table in sublist]

# Define the extract_text_from_image function
def extract_text_from_image(image):
    return pytesseract.image_to_string(image)

# Extract text from images using multiprocessing
def extract_images_and_text_from_pdf(pdf_path):
    images = convert_from_path(pdf_path)
    
    with mp.Pool(mp.cpu_count()) as pool:
        texts = pool.map(extract_text_from_image, images)
    
    return images, texts

# Load PDF using PyPDFLoader
pdf_path = "C:/Users/Talent2/Desktop/ness/employee_details.pdf"
text_from_pdf = extract_text_from_pdf(pdf_path)
tables_from_pdf = extract_tables_from_pdf(pdf_path)
images_from_pdf, image_texts_from_pdf = extract_images_and_text_from_pdf(pdf_path)

# Preprocess extracted data
cleaned_text = text_from_pdf
cleaned_tables = tables_from_pdf
cleaned_image_texts = image_texts_from_pdf

# Create Document objects with metadata
documents = [
    Document(page_content=cleaned_text, metadata={"type": "text"}),
    *[
        Document(page_content=str(table), metadata={"type": "table"})
        for table in cleaned_tables
    ],
    *[
        Document(page_content=image_text, metadata={"type": "image_text"})
        for image_text in cleaned_image_texts
    ]
]

# ChromaDB setup
key = os.getenv('OPENAPI_KEY')
embedding = OpenAIEmbeddings(api_key=key)
text_contents = [doc.page_content for doc in documents]
text_contents = [str(text) for text in text_contents]

# Create embeddings in batches
batch_size = 10  # Adjust batch size based on your use case
batches = [text_contents[i:i + batch_size] for i in range(0, len(text_contents), batch_size)]
embeddings = []
for batch in batches:
    embeddings.extend(embedding.embed_texts(batch))

# Create vector database from embeddings
vectordb = Chroma.from_embeddings(embeddings=embeddings, texts=text_contents)

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
