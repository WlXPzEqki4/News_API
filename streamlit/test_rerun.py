import streamlit as st

st.write("Streamlit version (detected by code):", st.__version__)
if st.button("Rerun Test"):
    st.experimental_rerun()
