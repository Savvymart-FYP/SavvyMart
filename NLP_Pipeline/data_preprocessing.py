import pandas as pd
import re
import string
import contractions 
import nltk
from nltk.corpus import stopwords
from joblib import Parallel, delayed
from tqdm import tqdm # Progress bar dekhne ke liye
import pkg_resources
from symspellpy import SymSpell, Verbosity

df = pd.read_csv(r'E:\SavvyMart\Model_Training\fake_reviews_dataset.csv')

#---------This line converted the review column into lower case letters---------

df['text_'] = df['text_'].str.lower()   

#---------This function will remove all the html tags etc from scrapping---------

def remove_html_tags(text):   
    pattern = re.compile('!<.*?>')
    return pattern.sub(r'', text)

df['text_'] = df['text_'].fillna("")

df['text_'] = df['text_'].apply(remove_html_tags)

#---------This function will remove all the links etc ---------
def remove_url(text):
    pattern = re.compile(r'https?://\S+|www\.\S+')
    return pattern.sub(r'', text)

df['text_'] = df['text_'].apply(remove_url)

#---------This function will remove all the Punctuations etc ---------

def remove_punc(text):
    return text.translate(str.maketrans('', '', string.punctuation))

df['text_'] = df['text_'].apply(remove_punc)


#---------This function will remove all the chatwords etc ---------

def chat_conversion(text):
    return contractions.fix(text)

df['text_'] = df['text_'].apply(chat_conversion)

#---------This function will remove all the stopwords etc ---------

# 1. Download and Cache Stopwords (Sirf ek baar)
nltk.download('stopwords')
stop_words_set = set(stopwords.words('english')) 

def remove_stopwords_fast(text):
    if not isinstance(text, str):
        return ""
    # Set check is O(1) complexity, List check is O(n)
    return " ".join([word for word in text.split() if word.lower() not in stop_words_set])

# 2. Parallel Processing Function
def parallel_process(df_column, func):
    # n_jobs=-1 matlab saare CPU cores use karo
    # delayed(func)(x) har row ko alag core pe bhejta hai
    results = Parallel(n_jobs=-1)(delayed(func)(x) for x in tqdm(df_column))
    return results

# 3. Run it!
df['text_'] = parallel_process(df['text_'], remove_stopwords_fast)




print(df.head())

df.to_csv(r'E:\SavvyMart\Model_Training\fake_reviews_dataset.csv', index=False)