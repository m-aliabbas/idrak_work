# Importing Libs
#
import streamlit as st
import dashboard.pages as p
import dashboard.file_loader_page as load_page

# #--------------------------Basic Page Config--------------------------------
#
st.set_page_config(layout="wide",page_title='Idrak Classification Dashboard')

    
# # ------------------------Defining Layout of Page ------------------------
# 
header = st.container()
main = st.container()
footer = st.container()

# #--------------------------- Adding Logo -----------------------------------
#
with header:
    _,c=st.columns((3,1))

    with c:
        st.image('idrak_logo.png',width=200)
        
# #--------------------------- Adding Pages ----------------------------------
#
