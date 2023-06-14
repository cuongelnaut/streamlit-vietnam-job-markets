import streamlit as st
import pandas as pd
# Title
st.title("Thông tin tuyển dụng")

title_job = st.text_input("Tiêu đề:", placeholder = 'Title')

title_job = st.text_input("Công ty:", placeholder = 'Company name')




lst_city = pd.read_csv('../data/provinces.csv').Provinces

city = st.multiselect("Vị trí: ", lst_city)

min_year, max_year = st.slider('Kinh nghiệm: (theo năm)', 0, 50, (5, 15), key='exp_year')
st.write(min_year,' tới ',max_year, 'năm')

st.selectbox('Cấp độ ứng tuyển: ',('a','b','c','d','e'))
col1, col2 =st.columns(2)
with col1:
    min_sal = st.number_input("Mức lương tối thiểu: ")
with col2:
    max_sal = st.number_input("Mức lương tối đa: ")

input_desc = st.radio("Chọn phương thức nhập cho mô tả công việc:", ('Tải file', 'Nhập từ bàn phím'))
if input_desc == 'Tải file':
    uploaded_desc_file = st.file_uploader(label='Nhập file tại đây!' ,type=['txt', 'pdf','docs'], key = 'input_desc')

if input_desc == 'Nhập từ bàn phím':
    with st.form(key='my_desc_form'):
        raw_desc_text = st.text_area("Input your text", max_chars=None)
        submit_desc_button = st.form_submit_button("Hoàn tất")

input_request = st.radio("Chọn phương thức nhập yêu cầu:", ('Tải file', 'Nhập từ bàn phím'),index=1)
if input_request == 'Tải file':
    uploaded_rq_file = st.file_uploader(label='Nhập file tại đây!' ,type=['txt', 'pdf','docs'],key='input_request')

if input_request == 'Nhập từ bàn phím':
    raw_rq_text = st.text_area("Input your text", max_chars=None)

lst = ['option1', 'option2', 'option3', 'option4', 'option5', 'option6', 'option7']
tags = ['option1', 'option2', 'option3']

st.multiselect("Lĩnh vực", lst, default=tags)
st.button('Hoàn thành')
	
