import streamlit as st
import pandas as pd
import pygwalker as pg
import streamlit.components.v1 as components

df=st.session_state.dfFiltered
pg_html=pg.walk(df,return_html=True)
components.html(pg_html,width=1000,height=1000,scrolling=True)