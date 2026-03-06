# import pandas as pd
# import nltk
# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer

# # 1. Zaroori resources
# nltk.download('punkt')
# nltk.download('punkt_tab')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

# # Path set karein
# input_file = r'E:\SavvyMart\NLP_Pipeline\fake_reviews_dataset.csv'
# output_file = r'E:\SavvyMart\NLP_Pipeline\cleaned_fake_reviews.csv' # Nayi file ka naam

# # 2. Data Load karein
# df = pd.read_csv(input_file)

# # 3. Lemmatizer initialize
# lemmatizer = WordNetLemmatizer()

# # 4. Lemmatization Function
# def lemmatize_text(text):
#     if isinstance(text, str):
#         tokens = word_tokenize(text.lower())
#         # Lemmatize verbs (actioning -> action) aur nouns (actions -> action)
#         lemmatized_output = [lemmatizer.lemmatize(w, pos='v') for w in tokens]
#         # Dobara join karein tokens ko string banane ke liye
#         return " ".join(lemmatized_output)
#     return str(text)

# # 5. Column update karein
# print("Cleaning and Lemmatizing... Please wait.")
# df['text_'] = df['text_'].apply(lemmatize_text)

# # 6. NAYI FILE SAVE KAREIN
# # index=False taake extra columns na banein
# df.to_csv(output_file, index=False)

# print("-" * 30)
# print(f"Done! Cleaned data save ho gaya hai yahan: \n{output_file}")
# print("-" * 30)
# print(df[['text_']].head())

# # Optional: Processed data ko save karne ke liye
# # df.to_csv('cleaned_dataset.csv', index=False)
# # Ab aap is par TF-IDF ya CountVectorizer chala sakte hain


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

df = pd.read_csv(r'E:\SavvyMart\NLP_Pipeline\fake_reviews_dataset.csv')

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