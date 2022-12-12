#------------------------------- Adding the Libs -----------------------------

import streamlit as st


# -------------------------- The Pages Class Defination ---------------------
#
class Pages:
    
    '''
    
    This class will be responisible for Managing pages
    ---- Adding Pages
    ---- Removing Pages
    ---- Moving Forward or Backwards
    
    '''

    #-------------------- Init Method of Class -----------------------------
    #
    def __init__(self):
        
        '''
        It will Intialize the Logic for managing pages
        '''
        
        # number of total pages added
        self.total_pages = 0
        # an array of pages 
        self.page = []
        # session variable which will keep track of current visible page
        if 'current_page' not in st.session_state:
            st.session_state['current_page'] = 0

        if 't_text' not in st.session_state:
            st.session_state['t_text'] = 'None'

        # Header Container
        self.header = st.container()
        # body container
        self.body = st.container()



    #----------------------------- Function to add Page ----------------------
    #
    def add_page(self, page):
        
        '''
        Add a page to List of Pages 
        ---- Increment total Page with 1
        ---- Append the Page to Page list
        
        args: 
                page: a page object you wish to add
                
        '''
        
        self.total_page += 1
        self.pages.append(page)
    
    #----------------------------- Function to add Page ----------------------
    #
    def del_page(self, page):
        
        '''
        Delete a page to List of Pages 
        ---- Decrement total Page with 1
        ---- Remove the Page to Page list
        
        args: 
                page: a page object you wish to remove
                
        '''
        
        self.total_page -= 1
        self.pages.remove(page)

    #------------------------ Function to Move Next Page ---------------------
    #
    def move_next(self):
        
        '''
        Get the current session state and add 1 to it
        if returned value is less than total page 
        set current page to it
        '''
        
        new_page = st.session_state.current_page + 1

        if new_page < self.total_page:
            st.session_state.current_page = new_page

    #------------------------ Function to Move Prev Page ---------------------
    #
    def move_back(self):
        
        '''
        Get the current session state and subs 1 from it
        if returned value is greater than zero
        set current page to it
        '''
        
        new_page = st.session_state.current_page - 1
        if new_page >= 0:
            st.session_state.current_page = new_page
    
    #-------------------- Function to Move on specific  ---------------------
    #
    def move_to_page(self, page_no):
        
        '''
        This fuction move the page to specified page
        
        args:
        
            page_no: the number of page you want to move
            
        '''
        new_page = page_no
        if (new_page >= 0) and (new_page < self.total_page):
            st.session_state.current_page = new_page

    #-------------------- Function to Add Title   --------------------------
    #
    def setTitle(self, text):
        
        '''
        Function to set the title of page
        args: 
        text: title of page
        '''
        
        st.session_state.t_text = text
        
    #-------------------- Function to Show Title   --------------------------
    #
    def showTitle(self):
        '''
        Function will show the title of page using Streamlit
        
        '''
        if st.session_state.t_text != '':
            with self.header:
                st.title(st.session_state.t_text)

    #------------------- Function to Added Pages to Body
    def show(self):

        '''
        This function will show page into layout
        '''
        with self.body:
            self.pages[st.session_state.current_page]()
