import streamlit as st
st.set_page_config(
        page_title="Idrak Metrics Compare",
        page_icon="chart_with_downwards_trend",
        layout="wide",
    )
import glob
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ai_tools.idrak_model_compare import IdrakModelCompare
from ai_tools.constant import DATADIR,classifiers_meta,dataset_names,classifier_names
from ai_tools.components import *

make_head()

sidebar=st.sidebar
with sidebar:
    dataset = sidebar.selectbox(
        "Select a Dataset From List",
        dataset_names
    )
    classifer = sidebar.selectbox(
        "Select a Classifier From List",
        classifier_names
    )


model_name='tinybert_report'
#path of file having classifier report 
file_path='{}\\{}\\'.format(DATADIR,model_name)

imc=IdrakModelCompare(model_dir_path=file_path,dataset=dataset_names[dataset],classifier=classifier_names[classifer])
ax=imc.visualize()
st.plotly_chart(ax)