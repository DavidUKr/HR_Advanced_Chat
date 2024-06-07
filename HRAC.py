import streamlit as st
from streamlit_chat import message
import os
from openai import OpenAI

from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
# RetrievalQA
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader

from langchain.callbacks import get_openai_callback
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import HumanMessage, AIMessage

st.title("AI Fragment analyzer")
if 'user_input' not in st.session_state:
    st.session_state['user_input'] = []
 
if 'openai_response' not in st.session_state:
    st.session_state['openai_response'] = []
 
def get_text():
    input_text = st.text_input("Ask your question here", key="input")
    return input_text
 
#text fragment input
user_fragment=st.text_area("Paste the desired fragment here (max 10 lines)", key='user_fragment')

#quetion input
user_input = get_text()