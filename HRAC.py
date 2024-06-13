import streamlit as st
from streamlit_chat import message
import os
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.callbacks import get_openai_callback
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders.csv_loader import CSVLoader

import fitz  # PyMuPDF
import io
import pytesseract
from PIL import Image
import pdfplumber
import csv
import pandas as pd
from collections import defaultdict



#extract csvs from pdf

def extract_table_titles(pdf_path):
    # Deschide PDF-ul
    doc = fitz.open(pdf_path)
    table_titles = []
    title_frequencies = defaultdict(int)

    # Variabilă pentru a ține evidența rândurilor goale între titlurile de tabele
    blank_lines_count = 0
    
    # Parcurge fiecare pagină
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text = page.get_text("text")
        lines = text.split('\n')
        
        for line in lines:
            words = line.split()
            
            # Verifică dacă linia este goală
            if len(words) == 0:
                blank_lines_count += 1
            else:
                # Verifică dacă linia conține un singur cuvânt care începe cu literă mare
                if len(words) == 1 and words[0][0].isupper():
                    # Pentru primul titlu de tabel, nu este nevoie să verificăm numărul de rânduri goale
                    if not table_titles or blank_lines_count >= 2:
                        table_titles.append(words[0])
                        title_frequencies[words[0]] = 0
                    blank_lines_count = 0  # Resetează contorul de rânduri goale
                else:
                    # Resetează contorul de rânduri goale dacă întâlnește o linie care nu este goală sau nu este titlu de tabel
                    blank_lines_count = 0

    return table_titles, dict(title_frequencies)
    
def extract_images_from_pdf(pdf_path, output_folder):
    pdf_document = fitz.open(pdf_path)
    
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        image_list = page.get_images(full=True)

        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]

            image = Image.open(io.BytesIO(image_bytes))
            image_path = os.path.join(f"Imagine.jpg")
            image.save(image_path)

            print(f"Saved image: {image_path}")

    print("Image extraction complete.")
    img = Image.open('Imagine.jpg')
    text = pytesseract.image_to_string(img)
    print(text)


def normalize_header(header):
    """Normalizează header-ul eliminând spațiile și caracterele de nouă linie."""
   # header=header.rstrip('\n')
    # header.replace('\n',"").strip()
    for df in header.columns:
        df=df.replace('\n','').strip()
    return header#header.replace('\n','').strip()

def extract_tables_from_pdf(pdf_path, output_folder):
    table_titles, title_frequencies = extract_table_titles(pdf_path)
    index=0
    all_tables_df = pd.DataFrame()
    # Cuvinte cheie de verificat în antetul tabelului (normalizate)
    keywords = {"region_id", "country_id", "location_id", "job_id"}
    
    # Creează directorul de ieșire dacă nu există
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            tables = page.extract_tables()

            for table_index, table in enumerate(tables):
                # Creează DataFrame din tabel
                if len(table) > 1:
                    df = pd.DataFrame(table[1:], columns=table[0])
                else:
                    continue  # Sari peste tabelele fără date
                
                # Normalizează antetul
                headers = [header for header in df.columns] #{normalize_header(header) for header in df.columns} # [header for header in df.columns] 
                #df=normalize_header(df)
                #headers2=[header for header in df.columns] 
                print(f"Page {page_num + 1}, Table {table_index + 1} headers: {headers}")
                
                # Verifică dacă antetul conține toate cuvintele cheie
                ok = 0
                for keyword in keywords:
                    if keyword in headers:
                        ok = 1
                        break

                if ok == 1:
                    # Construiește calea fișierului folosind os.path.join
                    table_path = os.path.join(output_folder, f"{table_titles[index]}.csv")
                    index=index+1
                    aux_header=headers
                    #df2=df
                    df.to_csv(table_path, index=False)
                    if all_tables_df.empty:
                            all_tables_df = df
                    else:
                            all_tables_df = pd.concat([all_tables_df, df], ignore_index=True)


                else: 
                    previous_table_path = os.path.join(output_folder, f"{table_titles[index - 1]}.csv")
                    df_existent = pd.read_csv(previous_table_path)
                    df.columns = aux_header
                    df_existent = pd.concat([df_existent, df], ignore_index=True)
                    print(df_existent)
                    df_existent.to_csv(previous_table_path, index=False)
    

                
                print(f"Saved table: {table_path}, OK: {ok}")
   
    print("Table extraction complete.")
    
# Example usage
pdf_path = 'employee_details.pdf'  # Path to your PDF file
output_folder = 'extracted_content'  # Output folder to save images and tables

if not os.path.exists(output_folder):
    os.makedirs(output_folder)
# ok
extract_images_from_pdf(pdf_path, output_folder)
extract_tables_from_pdf(pdf_path, output_folder)

#load documents
loader = DirectoryLoader(path="./extracted_content", glob="*.csv", loader_cls=CSVLoader)
docs = loader.load()

# Get API access
key = os.getenv('OPENAPI_KEY')
embedding = OpenAIEmbeddings(api_key=key)

# ChromaDB setup
persist_directory = 'db'
vectordb = Chroma.from_documents(documents=docs, 
                                 embedding=embedding,
                                 persist_directory=persist_directory)
vectordb.persist()
vectordb = None    
vectordb = Chroma(persist_directory=persist_directory, 
                  embedding_function=embedding)

# RAG setup
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
qa = RetrievalQA.from_chain_type(
    llm, retriever=vectordb.as_retriever()
)

conversation_buf = ConversationChain(
    llm=llm,
    memory=ConversationBufferMemory()
)

def run_with_conversation_buffer(chain, query, conversation_buf):
    with get_openai_callback() as cb:
        response = chain.invoke(query)
        conversation_buf.memory.save_context({"input": query}, {"output": str(response['result'])})
    return response['result']

# Streamlit setup
st.title("Human Resources Advanced Companion")
if 'user_input' not in st.session_state:
    st.session_state['user_input'] = []

if 'openai_response' not in st.session_state:
    st.session_state['openai_response'] = []

def get_text():
    input_text = st.text_input("Ask your question here", key="input")
    return input_text

#select box implementation
client = OpenAI(api_key=os.getenv('OPENAPI_KEY'))

query="You are an HR servant and need to provide answers in this format: name1,name2,name3, ...,namen from this pdf:"+str(documents)+". Please list the employees"

def get_employees():
    result = qa.invoke(query)

    return result['result']
    
employees=get_employees().split(', ')

selected_option = st.selectbox('Select an employee to ask about:', employees)

if selected_option:
    query="Who is "+selected_option
    output = run_with_conversation_buffer(qa, query, conversation_buf)
    # Store the output
    st.session_state.openai_response.append(output)
    st.session_state.user_input.append(query)

# Question input
user_input = get_text()

if user_input:
    output = run_with_conversation_buffer(qa, user_input, conversation_buf)
    # Store the output
    st.session_state.openai_response.append(output)
    st.session_state.user_input.append(user_input)

if st.session_state['user_input']:
    for i in range(len(st.session_state['user_input']) - 1, -1, -1):
        # This function displays user input
        message(st.session_state["user_input"][i], key=str(i), avatar_style="icons")
        # This function displays OpenAI response
        message(st.session_state['openai_response'][i], avatar_style="miniavs", is_user=True, key=str(i) + 'data_by_user')
