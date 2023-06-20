import streamlit as st
from plots.visualizations import visualizations

my_plots = visualizations()
st.set_page_config(layout='wide')
# Title
st.title("Số năm kinh nghiệm tối thiểu trung bình cho từng vị trí")

my_plots.plot_mean_min_years_by_level()