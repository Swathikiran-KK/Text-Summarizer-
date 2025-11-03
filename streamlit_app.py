import streamlit as st

st.set_page_config(page_title="Summarizer Suite", page_icon="📝", layout="wide")

home = st.Page("pages/text_app.py", title="Summarize", icon="📝")
history = st.Page("pages/history.py", title="History", icon="🗂️")

nav = st.navigation([home, history], position="sidebar", expanded=True)
nav.run()

