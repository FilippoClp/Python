# =============================================================================
# DEBT VULNERABILITY OF THE STOXX 600 INDEX NON FINANCIAL COMPANIES
# =============================================================================

# DISCLAIMER:
# The code has to be run bucket by bucket, starting with this one.

import os as os                      # Connect to local filesystem
import pyodbc                        # Connect to remote cloud database
import pandas as pd                  # Import data in a pandas dataframe 
import datetime as dt                # Import time conversion library

os.chdir("P:\\ECB business areas\\DGE\\DMP\\MAY\\NFC vulnerability\\Python_Project")

def wavg(group, avg_name, weight_name):

    d = group[avg_name]
    w = group[weight_name]
    
    try:
        return (d * w).sum() / w.sum()
    except ZeroDivisionError:
        return d.mean()
    
def extract_query(sql_filename):
    
    cnxn = pyodbc.connect("DSN=DISC DP Impala 64bit; UID=; PWD=", autocommit=True)
    cursor = cnxn.cursor()

    query = open("queries/" + sql_filename + ".sql", "r") 
    df = pd.read_sql_query(query.read(), cnxn)

    cursor.close() #Close connection
    cnxn.close() 
    
    return df

# =============================================================================
# OPEN ORBIS DATABASES AND MERGE TOGETHER INTERIM AND FINAL 
# =============================================================================

df_orbis = extract_query("query_orbis_STOXX600")
df_orbis_interim = extract_query("query_orbis_STOXX600_interim")

df_orbis = df_orbis[df_orbis['r1'] == 1]
df_orbis_interim = df_orbis_interim[df_orbis_interim['r1'] == 1]
df = df_orbis.append(df_orbis_interim)

df = df.sort_values(by=['country', 'id', 'year_long'])
df.drop_duplicates(subset =["id", "year_long"], keep = 'first', inplace = True)
del df_orbis, df_orbis_interim

identifiers = df['id'].drop_duplicates() #Check if the available IDs are 248
del identifiers

df.to_csv("csv/STOXX600_interim.csv", sep = ',', index=False)

# =============================================================================
# OPEN PRE-PROCESSED ORBIS DATABASE AND PRODUCE BALANCED PANEL                                                    
# =============================================================================

df = pd.read_csv("csv/STOXX600_interim.csv", sep = ',', index_col = False)

df['date'] = ''
df['date_time'] = pd.to_datetime(df['year_long'], format='%Y%m%d', errors='ignore')
df = df[df['date_time'] >= dt.date(2007,3,1)] 

for date in df.date_time.drop_duplicates():
    if date.month >= 3 and date.month <= 5: df['date'][df['date_time']==date] = dt.date(date.year,3,31)
    elif date.month >= 6 and date.month <= 8: df['date'][df['date_time']==date] = dt.date(date.year,6,30)
    elif date.month >= 9 and date.month <= 11: df['date'][df['date_time']==date] = dt.date(date.year,9,30) 
    elif date.month == 12: df['date'][df['date_time']==date] = dt.date(date.year,12,31)
    elif date.month == 1 or date.month == 2: df['date'][df['date_time']==date] = dt.date(date.year-1,12,31)

df['date'] = pd.to_datetime(df['date'], errors='ignore')
identifiers = df['id'].drop_duplicates()

df_balanced = pd.DataFrame(pd.date_range(dt.date(2007,3,1), dt.date(2019,9,1) + pd.offsets.QuarterBegin(1), freq='Q').tolist(), columns=['date'])
df_balanced = pd.merge(df_balanced,df[df.id==identifiers[0]], on='date', how='left')
df_balanced = df_balanced.sort_values(by=['date'])
df_balanced = df_balanced.fillna(method='ffill')
df_balanced = df_balanced.fillna(method='backfill')

identifiers = identifiers.drop([0])

for bvdidnumber in identifiers:
    df_append = pd.DataFrame(pd.date_range(dt.date(2007,3,1), dt.date(2019,9,1) + pd.offsets.QuarterBegin(1), freq='Q').tolist(), columns=['date'])
    df_append = pd.merge(df_append, df[df['id']==bvdidnumber], on='date', how='left')
    df_append = df_append.sort_values(by=['date'])
    df_append = df_append.fillna(method='ffill')
    df_append = df_append.fillna(method='backfill')
    df_balanced = df_balanced.append(df_append, ignore_index=True)

df_balanced = df_balanced.drop(columns=['year_long', 'date_time'])    
del bvdidnumber, df, df_append, identifiers, date

df_balanced.to_csv("csv/STOXX600_balanced.csv", sep = ',', index=False)










# =============================================================================
# OPEN EXPECTED DEFAULT PROBABILITIES                                               
# =============================================================================

df = pd.read_csv("csv/db_CRP.csv", sep = ',', index_col = False)

df['date'] = pd.to_datetime(df['EDFDATE'], format='%d-%b-%y', errors='ignore')

# Export Monthly Grouping
df['month'] = df['date'].dt.strftime('%m/%Y')
df_grouped = df.groupby(['ISSUERISIN', 'month'], as_index=False).mean()
df_grouped['month'] = '01/' + df_grouped['month']
df_grouped = df_grouped.rename(columns={'month':'date'})

df_grouped.to_csv("csv/EDF.csv", sep = ',', index=False)

# =============================================================================
# OPEN EXPECTED DEFAULT PROBABILITIES                                               
# =============================================================================

df_balanced = pd.read_csv("csv/STOXX600_interim_balanced.csv", sep = ',', index_col = False)

identifiers = df_balanced['isin'].drop_duplicates()
db_EDF = df_grouped[df_grouped['ISSUERISIN'].isin(identifiers)]

db_EDF.ISSUERISIN.drop_duplicates()

dt.datetime.strptime(df['year_long'],'%Y%m%d') 
date = dt.datetime.strptime('20180301','%Y%m%d')

# =============================================================================
# OPEN IBOXX FROM DISC                                             
# =============================================================================

df_iboxx = extract_query("query_iboxx")
df_iboxx.to_csv("csv/db_iboxx.csv", sep = ',', index=False)

# =============================================================================
# OPEN IBOXX FROM LOCAL AND MERGE IT WITH CRP                                           
# =============================================================================
