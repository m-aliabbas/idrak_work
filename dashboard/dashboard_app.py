# Importing Libs
#
import fix_columns as fixcols
import file_loader_page as load_page
import analayze
import split 
import show_stats
import merge 
import pages as p
import evaluate
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
obj.add_page(show_stats.main)
obj.add_page(fixcols.main)
obj.add_page(split.main)
obj.add_page(merge.main)
obj.add_page(analayze.main)
obj.add_page(evaluate.main)
with footer:
    primaryColor = st.get_option("theme.primaryColor")
    s = f"""
    <style>
    div.stButton > button:first-child {{ border: 1px solid {primaryColor}; border-radius:20px 20px 20px 20px;
    height: 3em;width: 16em;font:12px "Helvet";}}
    <style>
    """
    st.markdown(s, unsafe_allow_html=True)
    
    c1,c2,c3,c4,c5,c6,c7 = st.columns(7)
    with c1:
        btn_folder = st.button('Load & Clean')
        if btn_folder:
            # obj.setTitle('Load the File')
            obj.move_to_page(0)
    with c2:
        btn_folder = st.button('Show Stats')
        if btn_folder:
            # obj.setTitle('Load the File')
            obj.move_to_page(1)
    with c3:
        btn_folder = st.button('Fix Cols')
        if btn_folder:
            # obj.setTitle('Load the File')
            obj.move_to_page(2)
    with c4:
        btn_folder = st.button('Split')
        if btn_folder:
            # obj.setTitle('Load the File')
            obj.move_to_page(3)
    with c5:
        btn_folder = st.button('Merge')
        if btn_folder:
            # obj.setTitle('Load the File')
            obj.move_to_page(4)
    with c6:
        btn_folder = st.button('Analayze')
        if btn_folder:
            # obj.setTitle('Load the File')
            obj.move_to_page(5)
    with c7:
        btn_folder = st.button('Evaluate')
        if btn_folder:
            # obj.setTitle('Load the File')
            obj.move_to_page(6)
# #--------------------------- Rendering Main --------------------------------
# 
with main:
    obj.show()