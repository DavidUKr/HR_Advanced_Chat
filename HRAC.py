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

# Load PDF
loader = PyPDFLoader('Employee-details-1.pdf')
documents = loader.load()

# Get API access
key = os.getenv('OPENAPI_KEY')
embedding = OpenAIEmbeddings(api_key=key)

# ChromaDB setup
persist_directory = 'db'
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
vectordb = vectordb = Chroma.from_documents(documents=texts, embedding=embedding)

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
