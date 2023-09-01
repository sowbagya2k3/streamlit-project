import matplotlib
import streamlit as st
import numpy as np
import pandas as pd
from st_aggrid import AgGrid,GridOptionsBuilder,DataReturnMode
from streamlit_sortables import sort_items
from streamlit_extras.dataframe_explorer import dataframe_explorer
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI
from pandasai.llm.starcoder import Starcoder
import matplotlib.pyplot as plt


st.set_page_config(layout="wide",initial_sidebar_state="collapsed")
st.header("Explotary Data Analisis !!")
upload,edit,pivot=st.tabs(["uploaded Data","clean Data","AI"])
#st.session_state.dfFiltered=None
def ulploaddata(sel):
    print("y")
           

with upload:

    Selection = st.selectbox("Data source",["CSV","XML","Excel","JSON","API","Database"])
    match Selection:
        case "CSV":
            col1,col2=st.columns(2)
            
            filename=col1.file_uploader("Select file",type="csv")
            delimeter=col2.text_input("Deleimeter",value=",")
            qualifier=col2.text_input("Text Qualifier",value='"')
            proces=col1.button("upload")
            if proces:
                if filename:
                    df=pd.read_csv(filename,delimiter=delimeter,quotechar=qualifier)
                    st.session_state.dfFiltered=df
            
               
        case "XML":
            col1,col2=st.columns(2)
            
            filename=col1.file_uploader("Select file",type="csv")
            delimeter=col2.text_input("Deleimeter",value=",")
            qualifier=col2.text_input("Text Qualifier",value='"')
            proces=col1.button("upload")
            if proces:
                if filename:
                    df=pd.read_xml(filename,delimiter=delimeter,quotechar=qualifier)
                    st.session_state.dfFiltered=df   
    
    dfu=st.session_state.dfFiltered
    #st.dataframe(df)
    st.session_state.dfFiltered = dataframe_explorer(st.session_state.dfFiltered)#, case=False)
    st.dataframe(st.session_state.dfFiltered, use_container_width=True)
            
            
with pivot:
    df=st.session_state.dfFiltered
    
    llm=OpenAI(api_token="sk-in2XSzc4RaB45kra25NAfcaT3BlbkFJrrkaL5IjfJQA8XRW1rySAC")
    
    pandas_ai = PandasAI(llm)
    #import matplotlib
    matplotlib.use('TkAgg')
    prompt=st.text_area("enter your quetion here")
    if prompt is not None:
        LOAD=st.button("LOAD")
        if LOAD:
            returnVal=pandas_ai(df, prompt=prompt)
            print(returnVal)
            
            #plt.hist(Val)
            #st.pyplot(plt)

            
with edit:
    pde=st.session_state.dfFiltered
    if pde is not None:
        c1,c2,c3=st.columns(3)
        with st.container():
            c1.text("Drop na Values")
            selected=c2.multiselect("select columns",options=pde.columns.to_list())
            dropna=c3.button("Apply")
            if dropna:
                pde.dropna(subset=selected)
        st.session_state.dfFiltered=st.data_editor(pde)
