import matplotlib
import streamlit as st
import numpy as np
import pandas as pd

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
           
Selection = st.sidebar.selectbox("Data source",["CSV","XML","Excel","JSON","API","Database"])
with upload:
 
    match Selection:
        case "CSV":
            with st.form("CSV"):
                col1,col2=st.columns(2)
            
                filename=col1.file_uploader("Select file",type="csv")
                delimeter=col2.text_input("Deleimeter",value=",")
                qualifier=col2.text_input("Text Qualifier",value='"')
                proces=st.form_submit_button("upload")
                if proces:
                    if filename:
                        df=pd.read_csv(filename,delimiter=delimeter,quotechar=qualifier)
                        st.session_state.dfFiltered=df
            
               
        case "XML":
            col1,col2=st.columns(2)
            
            filename=col1.file_uploader("Select file",type="xml")
           
            proces=col1.button("upload")
            if proces:
                if filename:
                    df=pd.read_xml(filename)
                    st.session_state.dfFiltered=df   
        case "JSON":
            col1,col2=st.columns(2)
            
            filename=col1.file_uploader("Select file",type="json")
            
            proces=col1.button("upload")
            if proces:
                if filename:
                    df=pd.read_json(filename)
                    st.session_state.dfFiltered=df
        case "Excel":
            col1,col2=st.columns(2)
            
            filename=col1.file_uploader("Select file",type="xlsx")
            sheetName=col2.text_input("eneter sheet name")
            proces=col1.button("upload")
            if proces:
                if filename:
                    df=pd.read_excel(filename,sheet_name=sheetName)
                    st.session_state.dfFiltered=df
        case "API":
            import requests
            from requests.auth import HTTPDigestAuth
            import json
            url=st.text_input("Enter the URL")
            isauth=st.radio("Authendication",["Credentails","API Key"])

            if isauth=="Credentails":
                user=st.text_input("Eneter User name")
                password=st.text_input("Enter Password",type='password')
            if isauth=="API Key":
                user=st.text_input("Eneter API Key")
               
            process=st.button("get data")
            if process:
                if isauth=="Credentails":
                    auth = HTTPDigestAuth(user, password)
                    res=requests.get(url=url,auth=auth)
                elif isauth=="Credentails":
                    headers = {'Authorization': '[api_key]',}
                    res=requests.get(url=url,headers=headers)
                else:
                    res=requests.get(url=url)
                if(res.status_code==200):
                    if "json" in res.headers["Content-Type"]:
                        j=json.load(res.json)
                        df=pd.read_json(j)
                        st.session_state.dfFiltered=df
                

           
            

    if 'dfFiltered' in st.session_state:
        dfu=st.session_state.dfFiltered
        #st.dataframe(df)
        st.session_state.dfFiltered = dataframe_explorer(st.session_state.dfFiltered)#, case=False)
        st.dataframe(st.session_state.dfFiltered, use_container_width=True)
            
            
with pivot:
    if "dfFiltered" in st.session_state:
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
    if "dfFiltered" in st.session_state:
        pde=st.session_state.dfFiltered
        if pde is not None:
            c2,c3=st.columns(spec=[0.7,0.3])
            with st.container():
                
                selected=c2.multiselect("Drop na Values",options=pde.columns.to_list(),)
                dropna=c3.button("Apply")
                if dropna:
                    pde.dropna(subset=selected,inplace=True)
            with st.container():
                
                drop=c2.multiselect("drop duplicates",options=pde.columns.to_list(),key="drop")
                dropdup=c3.button("Apply",key="dropb")
                if dropdup:
                    pde.drop_duplicates(drop,inplace=True)

            st.session_state.dfFiltered=st.data_editor(pde)
