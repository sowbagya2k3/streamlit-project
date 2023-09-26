import pandas as pd

import streamlit as st
from ydata_profiling import ProfileReport
if 'dfFiltered' in st.session_state:
    df=st.session_state.dfFiltered
    profile = ProfileReport(df, title="Pandas Profiling Report")
    with st.spinner("Generating Report....\nplease wait...."):
    
        st.components.v1.html(profile.to_html(), width=1000, height=1200, scrolling=True)    