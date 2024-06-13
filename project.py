import streamlit as st
from streamlit_navigation_bar import st_navbar
import os
import dashboard
import chatbot
import profiles

# title_html = """
# <div style="position: absolute; top: 0; left: 0; z-index: 1; padding: 10px;">
#     <img src="https://emojicdn.elk.sh/👥" width="30" height="30" style="margin-right: 10px;">
#     <h1 style="display: inline; color: white;">HRBot</h1>
# </div>
# """

st.set_page_config(
    page_title="HRBot",
    page_icon="👥", 
    initial_sidebar_state="collapsed",
)

pages = ["Dashboard", "Chatbot", "Profile"]
styles = {
    "nav": {
        "background-color": "rgb(237, 131, 230, 1.5)",
        "z-index": "0",
    },
    "div": {
        "max-width": "40rem",
    },
    "span": {
        "border-radius": "0.2rem",
        "color": "rgb(49, 51, 63)",
        "margin": "0 0.125rem",
        "padding": "0.4375rem 0.625rem",
    },
    "active": {
        "background-color": "rgba(255, 255, 255, 0.25)",
    },
    "hover": {
        "background-color": "rgba(255, 255, 255, 0.35)",
    },
}

page = st_navbar(pages, styles=styles)

# Embed the title HTML
#st.markdown(title_html, unsafe_allow_html=True)

with st.sidebar:
    st.title("Settings")

if page == "Dashboard":
    dashboard.show_dashboard()
elif page == "Chatbot": 
    chatbot.show_chatbot()
elif page == "Profile": 
    profiles.show_profiles()
