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
from langchain.document_loaders import DataFrameLoader
from langchain.schema import Document

import fitz  # PyMuPDF
import io
import pytesseract
from PIL import Image
import pdfplumber
import csv
import pandas as pd
from collections import defaultdict

#UI
def show_chatbot():

    def set_page_bg_color(color):
        page_bg_style = f'''
        <style>
        .stApp {{
            background-color: {color};
        }}
        </style>
        '''
        st.markdown(page_bg_style, unsafe_allow_html=True)

    set_page_bg_color("#FFFFFF")        

    @st.cache_data
    def get_base64_of_bin_file(bin_file):
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()

    def image_with_position(png_file, top, left, width, height, hyperlink=None):
        bin_str = get_base64_of_bin_file(png_file)
        image_html = f'''
        <div class="image-container" style="position: relative; margin-top: {top}px; margin-left: {left}px;">
            <img src="data:image/png;base64,{bin_str}" style="width: {width}px; height: {height}px;">
        </div>
        '''
        st.markdown(image_html, unsafe_allow_html=True)
        if hyperlink:
            button_html = f'''
            <div class="button-container" style="position: relative; margin-top: -80px; margin-left: 520px;">
                <a href="{hyperlink}" target="_blank">
                    <button style="background-color: #ED82E6; color: white; border: none; padding: 10px 20px; cursor: pointer;">View Document</button>
                </a>
            </div>
            '''
            st.markdown(button_html, unsafe_allow_html=True)

    def image_with_position_border(png_file, top, left, width, height, style, radius, color, b, hyperlink=None):
        bin_str = get_base64_of_bin_file(png_file)
        image_html = f'''
        <style>
        .bordered-image {{
            border-style: {style};
            border-radius: {radius}px;
            border-color:  {color};
            border-width: {b}px;
            width: {width}px;
            height: {height}px;
            margin-top: {top}px;
            margin-left: {left}px;
        }}
        </style>
        <div>
            <img src="data:image/png;base64,{bin_str}" class="bordered-image">
        </div>
        '''
        st.markdown(image_html, unsafe_allow_html=True)
        if hyperlink:
            button_html = f'''
            <div class="button-container" style="position: relative; margin-top: -80px; margin-left: 520px;">
                <a href="{hyperlink}" target="_blank">
                    <button style="background-color: #ED82E6; color: white; border: none; padding: 10px 20px; cursor: pointer;">View Document</button>
                </a>
            </div>
            '''
            st.markdown(button_html, unsafe_allow_html=True)
        
        image_with_position_border('./static/images/enhance.jpg', top=0, left=0, width=400, height=493, style='solid', radius=10, color='black', b=2)

    #######################################################################

#extract csvs from pdf
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Modificați calea după locația instalării Tesseract

def preprocess_image(image_path):
    # Citește imaginea
    image = cv2.imread(image_path)
    
    # Verifică dacă imaginea a fost încărcată corect
    if image is None:
        raise FileNotFoundError(f"Imaginea nu a putut fi găsită la calea specificată: {image_path}")
    
    # Convertirea imaginii în nuanțe de gri
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Inversarea imaginii
    inverted = cv2.bitwise_not(gray)
    
    # Aplicarea unui filtru de umbrire pentru a îmbunătăți contrastul
    _, thresholded = cv2.threshold(inverted, 150, 255, cv2.THRESH_BINARY)
    
    # Salvarea imaginii preprocesate pentru verificare (opțional)
    preprocessed_path = 'preprocessed_image.png'
    cv2.imwrite(preprocessed_path, thresholded)
    print(f"Imagine preprocesată salvată la: {preprocessed_path}")
    
    return thresholded

def extract_text_from_image(image_path):
    try:
        # Preprocesarea imaginii
        preprocessed_image = preprocess_image(image_path)
        
        # Convertirea imaginii preprocesate la un format compatibil cu PIL
        pil_image = Image.fromarray(preprocessed_image)
        
        # Utilizarea Tesseract pentru a extrage textul
        text = pytesseract.image_to_string(pil_image)
        return text
    except Exception as e:
        print(f"Eroare la extragerea textului: {e}")
        return ""


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
    index=0
    
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        image_list = page.get_images(full=True)

        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]

            image = Image.open(io.BytesIO(image_bytes))
            image_path = os.path.join(output_folder,f"Image{index + 1}.jpg")
            index=index+1
            image.save(image_path)

            print(f"Saved image: {image_path}")
            

    print("Image extraction complete.")


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
                    df.loc[-1] = df.columns  # Adaugă antetul inițial ca prima linie
                    df.index = df.index + 1  # Mută toate indexurile în jos
                    df = df.sort_index()  
                    df.columns = aux_header
                    df_existent = pd.concat([df_existent, df], ignore_index=True)
                    #print(df_existent)
                    df_existent.to_csv(previous_table_path, index=False)
                    
    

                
                print(f"Saved table: {table_path}, OK: {ok}")
   
    print("Table extraction complete.")
    
# Example usage
pdf_path = 'Employee-details-1.pdf'  # Path to your PDF file
output_folder = 'extracted_content'  # Output folder to save images and tables

if not os.path.exists(output_folder):
    os.makedirs(output_folder)
# ok
extract_images_from_pdf(pdf_path, output_folder)
extract_tables_from_pdf(pdf_path, output_folder)

image_path = r"extracted_images\Image1.jpg"

try:
    extracted_text = extract_text_from_image(image_path)
    extracted_t="This info is about an employee, just like the oothers: " + extracted_text
    #print("Extracted Text:", extracted_text)
except FileNotFoundError as e:
    print(e)

documents = [Document(page_content=extracted_t, metadata={"source": "Image1.jpg"})]

path = r"extracted_content"
countries_df = pd.read_csv(os.path.join(path, 'Countries.csv'))
departments_df = pd.read_csv(os.path.join(path, 'Departments.csv'))
employees_df = pd.read_csv(os.path.join(path, 'Employees.csv'))
jobs_df = pd.read_csv(os.path.join(path, 'Jobs.csv'))
locations_df = pd.read_csv(os.path.join(path, 'Locations.csv'))
regions_df = pd.read_csv(os.path.join(path, 'Regions.csv'))
p1 = r"extracted_content/Countries.csv"
p2 = r"extracted_content/Departments.csv"
p3 = r"extracted_content/Employees.csv"
p4 = r"extracted_content/Jobs.csv"
p5 = r"extracted_content/Locations.csv"
p6 = r"extracted_content/Regions.csv"
merged_df = pd.merge(regions_df, countries_df, on='region_id')
merged_df = pd.merge(merged_df, locations_df, on='country_id')
merged_df = pd.merge(merged_df, departments_df, on='location_id')
merged_df = pd.merge(merged_df, employees_df, on='department_id')
merged_df = pd.merge(merged_df, employees_df, on='job_id')


# Print columns in merged_df to verify 'text' column existence
#print("Columns in merged_df:", merged_df.columns)

# Access 'text' column
merged_df['text'] = merged_df.astype(str).apply(' '.join, axis=1)
text_column = merged_df['text']

loader = DataFrameLoader(data_frame=merged_df, page_content_column='text')
documents = documents + loader.load()

# Create OpenAIEmbeddings instance
api_key = os.getenv('OPENAI_API_KEY')  # Replace with your OpenAI API key
embedding = OpenAIEmbeddings(api_key=api_key)

# ChromaDB setup
persist_directory = 'chroma_db'
vectordb = Chroma.from_documents(
    documents=documents,
    embedding=embedding,
    persist_directory=persist_directory
)
vectordb.persist()  # Persist the vector database to disk

# RAG setup
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
qa = RetrievalQA.from_chain_type(llm, retriever=vectordb.as_retriever())

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

# loader=PyPDFLoader('Employee-details-1.pdf')
# documents=loader.load()

# query="You are an HR servant and need to provide answers in this format: name1,name2,name3, ...,namen from this pdf:"+str(documents)+". Please list the employees in alphabetical order"

# def get_employees():
#     result = qa.invoke(query)

#     return result['result']
    
# employees=get_employees().split(', ')

# selected_option = st.selectbox('Select an employee to ask about:', employees)

# if selected_option:
#     query="Who is "+selected_option
#     output = run_with_conversation_buffer(qa, query, conversation_buf)
#     # Store the output
#     st.session_state.openai_response.append(output)
#     st.session_state.user_input.append(query)

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
