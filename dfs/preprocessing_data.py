import pandas as pd
import numpy as np
from googletrans import Translator
from os import listdir

df = pd.read_csv('drive/MyDrive/Public/DS102 - Machine Learning/data/job-markets-vn.csv')

detector = Translator()

def is_vietnamese(text):
    dec_lan = detector.detect(text)
    return dec_lan.lang == 'vi'