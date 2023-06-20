import streamlit as st
from plots.visualizations import visualizations

my_plots = visualizations()
st.set_page_config(layout='wide')
# Title
st.title("Mức lương trung bình theo theo số năm kinh nghiệm")

my_plots.plot_mean_salary_industry_by_yoe()