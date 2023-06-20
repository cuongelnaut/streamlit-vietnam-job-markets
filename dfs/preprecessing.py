
from googletrans import Translator
from vncorenlp import VnCoreNLP
from itertools import chain
import re
from unidecode import unidecode

def is_vietnamese(text):
    detector = Translator()
    dec_lan = detector.detect(text)
    return dec_lan.lang == 'vi'
    
with open('./data/vietnamese-stopwords.txt', 'r',encoding='utf-8') as file:
    vi_stopwords = file.readlines()
vi_stopwords = [sw.strip() for sw in vi_stopwords if sw.strip()]

def is_word(element):
    pattern = r'^\w+$'
    return re.match(pattern, element) is not None
# Remove special characters & stopwords
def remove_characters(txt_data):
    cleaned = []
    for token in txt_data:
        if token not in vi_stopwords:
            # if token.replace('_', '').isalpha():
            if is_word(unidecode(token)):
                # cleaned_token = re.sub(r"[-()\"#/@;:<>{}`+=~|!?,]", "", token)
                # cleaned.append(cleaned_token)
                cleaned.append(token)
    return cleaned

def preprocess(txt):
    # Lower case
    txt = is_vietnamese(txt)
    txt = txt.lower()
    # Tokenization
    txt = VnCoreNLP.tokenize(txt)
    txt = list(chain.from_iterable(txt))
    # Special character removal
    txt = remove_characters(txt)
    txt = list(filter(lambda x: x != '', txt))

    txt = ' '.join(txt).replace('_', ' ')
    return txt.strip()
def extract_features(txt):
    pass



