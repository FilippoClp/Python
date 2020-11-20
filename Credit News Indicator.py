import os as os                                                                # Connect to local filesystem
import pandas as pd
import string as st
import eikon as ek
import datetime as dt
import sklearn as sk 
import getFAMEData as fame
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

q = '(\\"loan\\" OR \\"loans\\" OR \\"guarantee\\" OR \\"guaranteed\\" OR \\"repayment\\" OR \\"repayments\\" OR \\"credit line\\" OR \\"credit facility\\") AND Topic:EZC'
days = pd.date_range(start='7-31-2020', end=dt.date.today(), freq ='1D').to_list()
#days = pd.date_range(start='7-28-2020', end=dt.date.today(), freq ='1D').to_list()
days = [day.strftime('%d-%m-%Y') for day in days]
headlines = ek.get_news_headlines(query=q, date_to=dt.date.today(), count=100)

for i in range(1,len(days)):
    df = ek.get_news_headlines(query=q, date_from=days[i-1], date_to=days[i], count=100)
    headlines = headlines.append(df, ignore_index=True)
    print ('Updating day: '+str(i))
    
headlines.drop_duplicates(keep = 'first', inplace = True) 
headlines.drop(columns=['sourceCode'], inplace = True) 
del i, q, df, days

stories = [ek.get_news_story(story) for story in headlines['storyId']]
stories = [html.html2text(story) for story in stories]
headlines['story'] = stories
del stories
headlines.to_csv("csv/stories_3007.csv", sep = ',', index=False)

# =============================================================================
# APPEND CURRENT DATABASE TO THE EXISTING ONE
# =============================================================================

headlines_old = pd.read_csv("csv/extended.csv", sep = ',', index_col = False)
headlines = pd.read_csv("csv/stories_temp.csv", sep = ',', index_col = False)

headlines = headlines.append(headlines_old, ignore_index=True, sort=True)
headlines.drop_duplicates(subset=['storyId'], keep = 'first', inplace = True)
headlines.to_csv("csv/extended.csv", sep = ',', index=False) 
del headlines_old

# =============================================================================
# BUILD THE INDICATOR - WEEKLY
# =============================================================================   

headlines = pd.read_csv("csv/extended.csv", sep = ',', index_col = False)
headlines['clean'] = [' '.join(clean_text(story)) for story in headlines['story']]
headlines['date'] = [date[0:10] for date in headlines['versionCreated']]
headlines['date']= pd.to_datetime(headlines['date'], errors='coerce')
headlines.drop(columns=['versionCreated'], inplace = True) 
headlines['week'] = headlines['date'].dt.strftime('%Y-%W')

kw_dict = pd.read_csv("csv/dict.csv", sep = ',', index_col = False)

headlines['deter'] = 0
headlines['improv'] = 0

for index, row in headlines.iterrows():
    for word in kw_dict[kw_dict['deterior'].notnull()]['deterior']:
        if row['clean'].find(word) != -1:
            headlines.loc[index, 'deter'] -= 1
    for word in kw_dict[kw_dict['improv'].notnull()]['improv']: 
        if row['clean'].find(word) != -1:
            headlines.loc[index, 'improv'] += 1
del index, row, word            

downs = headlines['deter'].groupby(headlines['week']).sum()/headlines['text'].groupby(headlines['week']).count()
ups = headlines['improv'].groupby(headlines['week']).sum()/headlines['text'].groupby(headlines['week']).count()
net = ups + downs

indicator = pd.DataFrame(list(zip(downs.index, ups, downs, net)), columns = ['date','ups', 'downs', 'chart'])
del kw_dict, net, ups, downs

# =============================================================================
# BUILD THE INDICATOR - MONTHLY
# =============================================================================   

indicator['datetime'] = pd.to_datetime(indicator.date.map(lambda x: str(x)+'-0'), format='%Y-%W-%w', errors='ignore')
indicator['month'] = indicator['datetime'].dt.strftime('%Y-%m')
indicator.drop(columns=['datetime'], inplace = True) 

downs = indicator['downs'].groupby(indicator['month']).min()
ups = indicator['ups'].groupby(indicator['month']).max()

indicator_m = pd.DataFrame(list(zip(downs.index, ups, downs)), columns = ['date','ups', 'downs'])
del ups, downs

#Calculate difference
indicator_m['ups_diff'] = indicator_m['ups'].diff()
indicator_m['downs_diff'] = indicator_m['downs'].diff()

#Standardize indicator series
indicator_m['ups_std'] = pd.Series
indicator_m['downs_std'] = pd.Series
indicator_m[['ups_std', 'downs_std']] = sk.preprocessing.StandardScaler().fit_transform(indicator_m[['ups_diff', 'downs_diff']])

#Download flows
flows = fame.getFAMEData('OTHER', '/global/disdatadg/esdb/restricted/mbs_prepub/prepub_bsi.db','BSI.M.U2.Y.U.A20T.A.4.U2.2240.Z01.E', startDate='2019-02-01')
indicator_m['flows'] = flows['OBS_VALUE']
del flows

indicator_m['flows_diff'] = indicator_m['flows'].diff()
indicator_m['flows_std'] = pd.Series
indicator_m[['flows_std']] = sk.preprocessing.StandardScaler().fit_transform(indicator_m[['flows_diff']])

#Estimate coefficients
X = indicator_m[['downs_diff','ups_diff']][1:-2]
Y = indicator_m['flows_diff'][1:-2]

regr = sk.linear_model.LinearRegression()
regr.fit(X,Y)

regr.intercept_
regr.coef_
del X, Y

indicator_m['intercept'] = regr.intercept_
indicator_m['coeff_downs'] = regr.coef_[0]
indicator_m['coeff_ups'] = regr.coef_[1]

res = res.reshape(-1,1)

res = regr.predict(X)
res_std = sk.preprocessing.StandardScaler().fit_transform(res)
