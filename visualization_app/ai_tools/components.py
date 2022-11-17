from streamlit_option_menu import option_menu
import pandas as pd
from PIL import Image
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.figure_factory as ff
@st.experimental_memo
def convert_df(df):
   return df.to_csv(index=False).encode('utf-8')
def show_panel2(df=pd.DataFrame(),selected_columns=[],labels=[],case_1=0):
    try:
        col1, col2 = st.columns((2,1))
        df_to_print=df[['cleaned_text','class','y_pred']]
        show_barchart(df,col=col2)
        col1.dataframe(df_to_print)
        csv = convert_df(df_to_print)
        st.download_button(
        "Press to Download",
        csv,
        "file.csv",
        "text/csv",
        key='download-csv'
        )
    except Exception as e:
        st.text('Error Nothing to Show in Graph')
def show_panel1(df=pd.DataFrame(),selected_columns=[],labels=[],case_1=0):
    selected_columns=selected_columns
    if case_1==1:
        col1, col2 = st.columns((1,1))
        cm=df['ConfusionMatrix'].values[0]
        selected_columns.remove('ConfusionMatrix')
        show_confusion_matrix(cm=cm,labels=list(labels.values()),col=col2)
        show_df(df=df,selected_columns=selected_columns,labels=labels,col=col1)
    elif case_1==2:
        col1, col2 = st.columns((1,1))
        selected_columns.remove('ClassificationReport')
        col2.text('\t'+df['ClassificationReport'].values[0])
        show_df(df=df,selected_columns=selected_columns,labels=labels,col=col1)
    elif case_1==3:
        col1, col2, col3 = st.columns((1,2,1))
        cm=df['ConfusionMatrix'].values[0]
        selected_columns.remove('ConfusionMatrix')
        show_confusion_matrix(cm=cm,labels=list(labels.values()),col=col2)
        col3.text('\t'+df['ClassificationReport'].values[0])
        show_df(df=df,selected_columns=selected_columns,labels=labels,col=col1)
    else:
        show_df(df=df,selected_columns=selected_columns,col=st)
def show_df(df=pd.DataFrame(),selected_columns=[],labels=[],col=''):
    df_to_show=df[selected_columns]
    col.dataframe(df_to_show)
def show_confusion_matrix(cm=[[]],labels=[],col=''):
    x,y=labels,labels
    z_text = [[str(y) for y in x] for x in cm]
    fig = ff.create_annotated_heatmap(cm, x=x, y=y, annotation_text=z_text)
    fig.update_layout(title_text='Confusion matrix',autosize=False,width=250,height=250,)
    fig.update_layout(margin=dict(t=50, l=20))
    # fig['data'][0]['showscale'] = True
    col.plotly_chart(fig, use_container_width=True)
    
col1, col2 = st.columns((1,2))
def streamlit_menu():
     option_menu(
            menu_title='Idrak Metrics Dashboard',  # required
            options=['Dashboard Home'],  # required
            # menu_icon="cast",  # optional
            default_index=0,  # optional
            orientation="horizontal",
        )


# def show_confusion_matrix(cm):
#     df_cm = pd.DataFrame(cm)
#     fig= plt.figure(figsize=(5, 5))
#     fig.suptitle('Confusion Matrix', fontsize=10)
#     # sns.heatmap(df_cm, annot=True, fmt='g')
#     col2.pyplot(fig)
    
def plot_history(df_history):
    fig = px.line(df_history[['train_acc','valid_acc']])
    fig.update_traces(textposition="bottom right")
    st.plotly_chart(fig)
def show_barchart(df_inv,col=''):
    # labels=list(df_inv.class_labels.unique())
    df_inv_groups=df_inv.groupby(by=['class_labels']).count()[['class']]
    df_inv_groups=df_inv_groups.rename(columns={'class':'class','class':'counts'})
    df_inv_groups=df_inv_groups.reset_index()
    bar_chart=px.bar(df_inv_groups,x='class_labels',y='counts',color_discrete_sequence =['#F63366']*len(df_inv_groups),
             template= 'plotly_white')
    bar_chart.update_layout(title_text='Invalid Prediction Frequency',autosize=True,width=250,height=250,)
    col.plotly_chart(bar_chart)
    
def make_head():
    col_1, col_2 = st.columns((1,2))
    col_1.image('logo.png',width=200)
    col_2.header('Idrak Model Performance Visualizer')
    streamlit_menu()
    
footer="""<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: black;
text-align: center;
}
</style>
<div class="footer">
<p>Developed with ❤ by <a style='display: block; text-align: center;' href="https://www.idrakai.com/" target="_blank">M. Ali Abbas ML Engr. Idrak AI</a></p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)
    
