import streamlit as st
from plots.visualizations import visualizations

my_plots = visualizations()
st.set_page_config(layout='wide')
# Title
st.title("Tương quan giữa mức lương và địa điểm làm việc")

my_plots.plot_mean_salary_by_provinces()