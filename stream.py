import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import nbformat
import plotly.express as px
st.set_page_config(page_title="Türkiye Enflasyon Tahmini")
tabs=["Yıllık Enflasyon","Aylık Enflasyon","Model Bazlı Tahmin"]
page=st.sidebar.radio("Sekmeler",tabs)
yıllıktahmin=pd.read_csv("yıllıktahmin.csv")
yıllıktahmin=yıllıktahmin.set_index(yıllıktahmin["Unnamed: 0"])
del yıllıktahmin["Unnamed: 0"]
yıllıktahmin=yıllıktahmin.rename_axis(["Tarih"])
aylık=pd.read_csv('aylık.csv')
aylık=aylık.set_index(aylık["Unnamed: 0"])
del aylık["Unnamed: 0"]
aylık=aylık.rename_axis(["Tarih"])
aylık.columns=["Aylık Enflasyon"]
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=yıllıktahmin.index[10:12],y=[61.94,60.84],mode='markers',name="Geçmiş Tahminler"))
fig1.add_trace(go.Scatter(x=yıllıktahmin.index[:12],y=yıllıktahmin["Ortalama"].iloc[:12],mode='lines',name="Enflasyon"))
fig1.add_trace(go.Scatter(x=yıllıktahmin.index[11:27],y=yıllıktahmin["Ortalama"].iloc[11:27],mode='lines',name="Tahmin"))
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=aylık.iloc[:118].index,y=aylık.iloc[:118,0],mode='lines',name="Aylık Enflasyon"))
fig2.add_trace(go.Scatter(x=aylık.iloc[117:].index,y=aylık.iloc[117:,0],mode='lines',name="Aylık Enflasyon Tahmini"))
if page=='Yıllık Enflasyon':
    st.markdown("<h1 style='text-align:center;'>Yıllık Enflasyon Tahmini</h1>",unsafe_allow_html=True)
    st.plotly_chart(fig1)
if page=='Aylık Enflasyon':
    st.markdown("<h1 style='text-align:center;'>Aylık Enflasyon Tahmini</h1>",unsafe_allow_html=True)
    st.plotly_chart(fig2)