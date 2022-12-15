# Importing Libs
#
import file_loader_page as load_page
import pages as p
import streamlit as st

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
        
# #--------------------------- Managing Session ------------------------------
#
obj = p.Pages()
st.session_state.flag = 1

# #--------------------------- Adding Pages ----------------------------------
#
obj.add_page(load_page.main)


with footer:
    primaryColor = st.get_option("theme.primaryColor")
    s = f"""
    <style>
    div.stButton > button:first-child {{ border: 1px solid {primaryColor}; border-radius:20px 20px 20px 20px;
    height: 3em;width: 24em;font:12px "Helvet";}}
    <style>
    """
    st.markdown(s, unsafe_allow_html=True)

    c1,c2,c3,c4,c5,c6,c7 = st.columns(7)
    with c1:
        btn_folder = st.button('Load & Spectate File')
        if btn_folder:
            # obj.setTitle('Load the File')
            obj.move_to_page(0)
    

# #--------------------------- Rendering Main --------------------------------
# 
with main:
    obj.show()