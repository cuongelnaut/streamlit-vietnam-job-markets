import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
from dfs.preprecessing import preprocess
from googletrans import Translator
from vncorenlp import VnCoreNLP
from itertools import chain
import re
from unidecode import unidecode

cities_lst = pd.read_csv('./data/provinces.csv').Provinces
industries_lst = pd.read_csv('./data/job_labels.csv')['0']
levels_lst = pd.read_csv('./data/levels.csv').name
dict_job = dict(industries_lst)

Path1 = './model/XLMBert_bGRU.h5'
Path2 = './model/XLMBert_bLSTM.h5'
Path3 = './model/XLMBert_customized.h5'
Path4 = './model/XLMBert_TextCNN.h5'

tags_index = []
tags = []
# Title
st.markdown("<h1 style='text-align: center; color: red;'>Thông tin tuyển dụng</h1>", unsafe_allow_html=True)
# st.title("Thông tin tuyển dụng", style="text-align: center")

st.write("Tiêu đề:")
title_job = st.text_input("Tiêu đề:", placeholder = 'Title', label_visibility = "collapsed")

comp_name_col, comp_city_col = st.columns([0.7,0.3])
with comp_name_col:
    st.write("Tên công ty:")
    comp_name = st.text_input("Tên công ty:", placeholder = 'Company name', label_visibility = "collapsed")
with comp_city_col:
    st.write("Địa chỉ: ")
    city = st.multiselect("Địa chỉ: ", cities_lst, label_visibility = "collapsed")

level_col, exp_years_col = st.columns([0.4,0.6])
with exp_years_col:
    st.write("Kinh nghiệm:")
    input_exp_years = st.radio("Kinh nghiệm:", ('Từ ... đến...', 'Không yêu cầu'), horizontal = True, label_visibility = "collapsed")
    if input_exp_years == 'Từ ... đến...':
        min_year, max_year = st.slider('Kinh nghiệm:', 0, 50, (5, 15), key='exp_year', label_visibility = "collapsed")
with level_col:
    st.write("Cấp độ: ")
    level = st.selectbox('Cấp độ: ',('a','b','c','d','e'), label_visibility = "collapsed")



st.write('Mức lương:')
min_sal_col, max_sal_col =st.columns(2)

with min_sal_col:
    min_sal = st.number_input("Tối thiểu: ", min_value = 1.0 , step = 0.1)
with max_sal_col:
    max_sal = st.number_input("Tối đa: ", min_value = min_sal, step = 0.1)

st.write("Mô tả công việc:")
input_desc = st.radio("Mô tả công việc:", ('Tải file', 'Nhập từ bàn phím'), horizontal = True, label_visibility = "collapsed")

if input_desc == 'Tải file':
    uploaded_desc_file = st.file_uploader("Input your text", type=['txt', 'pdf','docs'], key = 'input_desc', label_visibility = "collapsed")
if input_desc == 'Nhập từ bàn phím':
    with st.form(key='my_desc_form'):
        raw_desc_text = st.text_area("Input your text", label_visibility = "collapsed")
        submit_desc_button = st.form_submit_button("Hoàn tất")

st.write("Yêu cầu công việc:")
input_request = st.radio("Yêu cầu công việc:", ('Tải file', 'Nhập từ bàn phím'), index=1, horizontal = True, label_visibility = "collapsed")

if input_request == 'Tải file':
    uploaded_rq_file = st.file_uploader("Input your text", type=['txt', 'pdf','docs'],key='input_request', label_visibility = "collapsed")
if input_request == 'Nhập từ bàn phím':
    raw_rq_text = st.text_area("Input your text", label_visibility = "collapsed")
#Chọn model trainning
input_model = st.radio("Model:", ('bGRU', 'bLSTM','customized','TextCNN'), horizontal = True)
if input_model == 'bGRU':
    path = Path1
if input_model == 'bLSTM':
    path = Path2
if input_model == 'customized':
    path = Path3
if input_model == 'TextCNN':
    path = Path3  

model = load_model(path)
#get value predict
if input_desc == 'Tải file':
    if uploaded_desc_file:
        file = uploaded_desc_file.read().decode("utf-8")
        tags_index = model.predict(file)
if input_desc == 'Nhập từ bàn phím':
    if submit_desc_button:
        tags_index = model.predict(raw_desc_text)
        submit_desc_button = False

tags = [dict_job[key] for key in tags_index]
st.write("Lĩnh vực: ")
industry = st.multiselect("Lĩnh vực: ", industries_lst, default=tags, label_visibility = "collapsed")

finish_button = st.columns(4)
with finish_button[3]:
    st.button('Hoàn thành', type="primary")
	
