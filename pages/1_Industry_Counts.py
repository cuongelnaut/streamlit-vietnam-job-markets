import streamlit as st
from plots.visualizations import visualizations

my_plots = visualizations()
st.set_page_config(layout='wide')
# Title
st.title("Phân bố các ngành nghề")

# Plotting industry counts
my_plots.plot_industry_counts()