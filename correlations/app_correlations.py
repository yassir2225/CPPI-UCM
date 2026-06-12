import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="MASI–MBI Correlations", layout="wide")

st.title("Analyse des corrélations MASI–MBI")

with open("correlations/dashboard_correlations_1.html", "r", encoding="utf-8") as f:
    html_content = f.read()

components.html(html_content, height=900, scrolling=True)