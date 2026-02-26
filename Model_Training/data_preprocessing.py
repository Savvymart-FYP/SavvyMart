import pandas as pd
import re

df = pd.read_csv(r'E:\SavvyMart\Model_Training\fake_reviews_dataset.csv')

df['text_'] = df['text_'].str.lower()   #This line converted the review column into lower case letters


def remove_html_tags(text):   #This function will remove all the html tags etc from scrapping
    pattern = re.compile('!<.*?>')
    return pattern.sub(r'', text)

df['text_'] = df['text_'].apply(remove_html_tags)


print(df.head())