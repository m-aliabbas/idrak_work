import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)
from ai_tools.components import *

make_head()
st.write("# Welcome to Idrak Data Visulaizer! 👋")


st.markdown(
    """
    Please Select From Pages Section:
    - analyize_signle_model=> This App Will Make Reports for a Single Model Selected
    - analyize_multi_model=> This App will compare Different Model Trained on Same or different dataset
    """
)