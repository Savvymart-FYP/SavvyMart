import pandas as pd
import re

df = pd.read_csv(r'E:\SavvyMart\Model_Training\fake_reviews_dataset.csv')

df['text_'] = df['text_'].str.lower()   #This line converted the review column into lower case letters




print(df.head())