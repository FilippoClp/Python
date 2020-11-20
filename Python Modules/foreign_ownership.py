# =============================================================================
# INDEBTEDNESS OF THE FOREIGN SUBSIDIARIES IN THE EURO AREA
# =============================================================================

# DISCLAIMER:
# The code has to be run bucket by bucket, starting with this one.
# If you don't need to run the queries again, go directly to the bucket called
# "OPEN LOCAL COPY OF MERGED DATABASE"

import os as os                      # Connect to local filesystem
import pyodbc                        # Connect to remote cloud database
import pandas as pd                  # Import data in a pandas dataframe 
import numpy as np                   # Import statistical analysis library
import matplotlib.pyplot as plt      # Import plotting library

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
# OPEN ORBIS DATABASES FROM DISC OR LOCAL 
# =============================================================================

df_guo = extract_query("query_guo")
df_orbis = extract_query("query_orbis_U2") # use query_orbis_U3 for the massive database (run in STATA)

df_guo = pd.read_csv("csv/db_guo.csv", sep = ',', index_col = False)
df_orbis = pd.read_csv("csv/db_orbis.csv", sep = ',', index_col = False)

# =============================================================================
# CLEAN THE ORBIS OWNERSHIP/GUO DATABASE
# =============================================================================

# EXPLAINER:
# If the GUO is a phyisical person, Orbis records the GUO country as the nationality of the GUO owner and not as the country of 
# residence. Therefore, firms which are registered in countries outside of the Euro area by EU citizens are considered "foreigner".
# On the other end, firms registered in the Euro area by extra-EU citizens are considered "European". In a nutshell, only firms 
# whose legal entity is registered outside the Euro area are included in the sample below.

df_guo.drop_duplicates(subset =["bvdidnumber", "guobvdidnumber"], keep = 'first', inplace = True) #no duplicates CLEAN
df_guo.drop_duplicates(subset =["bvdidnumber"], keep = 'first', inplace = True) 

df_guo = df_guo[(df_guo.ishcountryisocode != "-") & (df_guo.guocountryisocode != "-")]
df_guo = df_guo[df_guo.guocountryisocode.str.len() == 2] #eliminate wrong column assignment. CONTACT DISC TEAM

df_eur = df_guo[df_guo.guocountryisocode.isin(['AT','BE','CY','DE','EE','ES','FI','FR','GR','IE','IT','LT','LU','LV','MT','NL','PT','SI','SK'])]
df_eur = df_eur[df_eur.guotype != "One or more named individuals or families"]

df_for = df_guo[~df_guo.guocountryisocode.isin(['AT','BE','CY','DE','EE','ES','FI','FR','GR','IE','IT','LT','LU','LV','MT','NL','PT','SI','SK'])]
df_for = df_for[df_for.guotype == "One or more named individuals or families"]
df_for = df_for[df_for.ishcountryisocode.isin(['AT','BE','CY','DE','EE','ES','FI','FR','GR','IE','IT','LT','LU','LV','MT','NL','PT','SI','SK'])]

df_guo=df_guo[~df_guo.bvdidnumber.isin(df_eur.bvdidnumber)] #removing firms owned by EU GUOs and extra-EU shareholders
df_guo=df_guo[~df_guo.bvdidnumber.isin(df_for.bvdidnumber)] #removing firms owned by extra-EU persons and registered in EU

del df_eur, df_for

#NOTE: Maybe having a look on those financial firms in LU owned by foreigners would not be a bad idea.

# =============================================================================
# CLEAN THE ORBIS FINANCIALS DATABASE
# =============================================================================

df_orbis.loans = df_orbis.loans.fillna(0)
df_orbis.ltdebt = df_orbis.ltdebt.fillna(0)

# =============================================================================
# MERGE THE TWO DATASETS AND CLEAN-UP THE MEMORY
# =============================================================================

df = df_orbis[df_orbis['id'].isin(df_guo.bvdidnumber)]
del df_orbis, df_guo

# =============================================================================
# OPEN LOCAL COPY OF MERGED DATABASE -> START HERE!
# =============================================================================

df = pd.read_csv("csv/db_new.csv", sep = ',', index_col = False)

# =============================================================================
# REMOVE DUPLICATES FROM PANEL DATA (MULTIPLE YEARS)
# =============================================================================

df['year'] = df.year_long.astype(str).str.slice(0, 4)
df.sort_values(['country', 'id', 'year_long'], ascending=False)
df.drop_duplicates(subset =["id", "year"], keep = 'first', inplace = True)
df = df.drop(columns=['year_long'])

# =============================================================================
# CREATE OTHER VARIABLES
# =============================================================================

df['tot_debt'] = df.loans + df.ltdebt
df['lev_assets'] = df.tot_debt/df.tot_assets

# =============================================================================
# GROUP AND PLOT
# =============================================================================

# count = df[df.country=="ES"].groupby('year').id.count() #DOVETE CARICARE STO CAZZO DI 2018 (e 2017 per la Germania..)

# df_IT = df[df.country=="IT"]
# df_IT.sort_values(['country', 'id', 'year'], ascending=False)
# df_IT['delta_lev'] = df_IT.groupby('id').diff().lev_assets
# df_IT = df_IT[(df_IT.year == "2015") | (df_IT.year == "2016") | (df_IT.year == "2017")]

 # df = df[df.lev_assets.notnull()]
 
df = df[(df.lev_assets>np.percentile(df.lev_assets,1)) & (df.lev_assets<np.percentile(df.lev_assets,99))]

lev_EA = df[df.year!='2018'].groupby('year').apply(wavg,"lev_assets", "tot_assets")
lev_DE = df[(df.country=="DE") & (df.year!='2018')].groupby('year').apply(wavg,"lev_assets", "tot_assets")
lev_ES = df[(df.country=="ES") & (df.year!='2018')].groupby('year').apply(wavg,"lev_assets", "tot_assets")
lev_FR = df[(df.country=="FR") & (df.year!='2018')].groupby('year').apply(wavg,"lev_assets", "tot_assets")
lev_IT = df[(df.country=="IT") & (df.year!='2018')].groupby('year').apply(wavg,"lev_assets", "tot_assets")

plt.plot(lev_EA, label="EA")
plt.plot(lev_DE, label="DE")
plt.plot(lev_ES, label="ES")
plt.plot(lev_FR, label="FR")
plt.plot(lev_IT, label="IT")
plt.legend()

lev_NL = df[(df.country=="NL") & (df.year!='2018')].groupby('year').apply(wavg,"lev_assets", "tot_assets")
lev_LU = df[(df.country=="LU") & (df.year!='2018')].groupby('year').apply(wavg,"lev_assets", "tot_assets")
lev_IE = df[(df.country=="IE") & (df.year!='2018')].groupby('year').apply(wavg,"lev_assets", "tot_assets")
lev_MT = df[(df.country=="MT") & (df.year!='2018')].groupby('year').apply(wavg,"lev_assets", "tot_assets")

plt.plot(lev_NL, label="NL")
plt.plot(lev_LU, label="LU")
plt.plot(lev_IE, label="IE")
plt.plot(lev_MT, label="MT")
plt.legend()