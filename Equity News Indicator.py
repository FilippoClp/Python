import os as os                                                                # Connect to local filesystem
import pandas as pd
import string as st
import eikon as ek
import datetime as dt
import re
import html2text as html
import nltk

# DEFINE FUNCTION THAT WILL CLEAN AND STEM WORDS
def clean_text(text):
    
    ps = nltk.PorterStemmer()
    stopwords = nltk.corpus.stopwords.words('english')
    
    text = "".join([word.lower() for word in text if word not in st.punctuation])
    tokens = re.split('\W+', text)
    text = [ps.stem(word) for word in tokens if word not in stopwords]
    return text

# Eikon API key (search for 'APPKEY' in Eikon)
ek.set_app_key('510c1a92eb4c4b01802c252bb8195a0e11e88194')
os.chdir("P:/ECB business areas/DGE/DMP/MAY/NFC vulnerability/Eikon Indicators/")

# =============================================================================
# CREATE DATABASE WITH DATE,HEADLINE AND STORIES
# =============================================================================

q = '(\\"Kapitalerhöhung\\" OR \\"Ampliación de capital\\" OR \\"Augmentation de capital\\" OR \\"Aumento di capitale\\") AND Topic:EZC'
headlines = ek.get_news_headlines(query=q, date_to='8-1-2020', count=100)

dataset = pd.read_csv("csv/equity.csv", sep = ',', index_col = False)

dataset = dataset.append(headlines, ignore_index=True)
dataset.drop_duplicates(subset=['storyId'], keep = 'first', inplace = True)

dataset.to_csv("csv/equity.csv", sep = ',', index=False) 

del q, headlines, dataset

# =============================================================================
# BUILD THE INDICATOR
# =============================================================================   

headlines = pd.read_csv("csv/equity.csv", sep = ',', index_col = False)
headlines['text'] = [text.lower() for text in headlines['text']]
headlines['date'] = [date[0:10] for date in headlines['versionCreated']]
headlines['date']= pd.to_datetime(headlines['date'], errors='coerce')
headlines.drop(columns=['versionCreated'], inplace = True) 
headlines['month'] = headlines['date'].dt.strftime('%Y-%m')

headlines['DE'] = 0
headlines['ES'] = 0
headlines['FR'] = 0
headlines['IT'] = 0

for index, row in headlines.iterrows():
    if row['text'].find("kapitalerhöhung") != -1:
            headlines.loc[index, 'DE'] += 1
    if row['text'].find("ampliación de capital") != -1:
            headlines.loc[index, 'ES'] += 1
    if row['text'].find("augmentation de capital") != -1:
            headlines.loc[index, 'FR'] += 1
    if row['text'].find("aumento di capitale") != -1:
            headlines.loc[index, 'IT'] += 1
del index, row                

DE = headlines['DE'].groupby(headlines['month']).sum()
ES = headlines['ES'].groupby(headlines['month']).sum()
FR = headlines['FR'].groupby(headlines['month']).sum()
IT = headlines['IT'].groupby(headlines['month']).sum()

indicator = pd.DataFrame(list(zip(DE.index, DE, ES, FR, IT)), columns = ['date','DE', 'ES', 'FR', 'IT'])
del DE, ES, FR, IT