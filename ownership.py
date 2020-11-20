# =============================================================================
# CREATE TABLES LINKING SUBSIDIARIES TO BANKERS AND ISSUERS
# =============================================================================

import os as os                      # Connect to local filesystem
import pyodbc                        # Connect to remote cloud database
import pandas as pd                  # Import data in a pandas dataframe 

os.chdir("P:/ECB business areas/DGE/DMP/MAY/NFC vulnerability/Ownership")


def extract_query(sql_filename):
    
    cnxn = pyodbc.connect("DSN=DISC DP Impala 64bit; UID=; PWD=", autocommit=True)
    cursor = cnxn.cursor()
    query = open("queries/" + sql_filename + ".sql", "r") 
    df = pd.read_sql_query(query.read(), cnxn)
    cursor.close() #Close connection
    cnxn.close() 
    
    return df


def extract_query_string(query):
    
   cnxn = pyodbc.connect("DSN=DISC DP Impala 64bit; UID=; PWD=", autocommit=True)
   cursor = cnxn.cursor()
   df = pd.read_sql_query(query, cnxn)
   cursor.close() 
   cnxn.close()  
    
   return df


def query_string(lei):
    
    lei = lei.dropna()
    lei = '"' + lei + '",'
    lei.iloc[-1] = lei.iloc[-1].replace(',',')')
    lei = lei.to_string(index=False)
    lei = lei.replace('\n', ' ')
    
    return lei

    
# =============================================================================
# CREATE TABLE FIRMS-BANKERS
# =============================================================================
 
# Run queries in Impala
guo = extract_query('guo_ish')
detail = extract_query('detail')
bnk = extract_query('bankers')

# Keep only the subsidiaries whose parents can be traced to the table of Orbis Detailed
df = guo[guo['guobvdidnumber'].isin(detail['bvdidnumber'])]
df2 = guo[guo['ishbvdidnumber'].isin(detail['bvdidnumber'])]
firms = df.append(df2)
firms.drop_duplicates(keep = 'first', inplace = True) 
del guo, df, df2

# Merge GUO and ISH to create one single corporate field
filter = (firms['guotype']!='Corporate') & (firms['ishtype']=='Corporate')
firms.loc[filter, 'guobvdidnumber'] = firms.loc[filter, 'ishbvdidnumber']
firms.loc[filter, 'guoname'] = firms.loc[filter, 'ishname']
firms.loc[filter, 'guocountryisocode'] = firms.loc[filter, 'ishcountryisocode']
firms.loc[filter, 'guotype'] = firms.loc[filter, 'ishtype']

firms = firms[firms['guotype']=='Corporate']
firms.drop_duplicates(subset=['bvdidnumber', 'guobvdidnumber'], keep = 'first', inplace = True) 
firms.drop(columns=['ishname', 'ishbvdidnumber', 'ishcountryisocode', 'ishtype', 'guotype'], inplace = True) 
del filter

# Merge with additional fields and finalize table
detail.rename(columns={"bvdidnumber": "guobvdidnumber"}, inplace=True)
firms = pd.merge(firms, detail, on='guobvdidnumber', how='left')
firms = pd.merge(firms, bnk, on='bvdidnumber', how='left')
firms = firms[['bvdidnumber', 'name_internat', 'countryisocode', 'lei', 'isin', 'guobvdidnumber', 'guoname', 'guocountryisocode', 'guolei', 'guoisin', 'bnkfullname', 'bnkbvdidnumber']]
del bnk, detail

# Export CSV
firms.to_csv("csv/firms.csv", sep = ',', index=False)


# =============================================================================
# CREATE TABLE FIRMS-ISSUERS
# =============================================================================

#------------------------------------------------------------------------------
firms = pd.read_csv("csv/firms.csv", sep = ',', index_col = False)
firms.drop(columns=['bnkfullname', 'bnkbvdidnumber'], inplace = True) 
firms.drop_duplicates(keep = 'first', inplace = True) 
#------------------------------------------------------------------------------

# Temporarily split the dataset into groups and firms
groups = firms[firms['bvdidnumber']==firms['guobvdidnumber']]
subs = firms[firms['bvdidnumber']!=firms['guobvdidnumber']]
del firms


# Create string to select individual group LEIs and merge it with CSDB, TAKE #1
lei = query_string(groups['guolei'].drop_duplicates())
with open('queries/csdb.sql', 'r') as file:
    query = file.read() + lei
csdb = extract_query_string(query)
del lei, query

# Merge the results with ORBIS, TAKE #1
csdb.rename(columns={'issuerexternalcode_lei': 'guolei'}, inplace=True)
df = pd.merge(groups, csdb, on='guolei', how='left')


# Create string to select individual group LEIs and merge it with CSDB, TAKE #2
lei = query_string(subs['lei'].drop_duplicates())
with open('queries/csdb.sql', 'r') as file:
    query = file.read() + lei
csdb = extract_query_string(query)
del lei, query

# Merge the results with ORBIS, TAKE #2
csdb.rename(columns={'issuerexternalcode_lei': 'lei'}, inplace=True)
df2 = pd.merge(subs, csdb, on='lei', how='left')

issuers = df.append(df2)
del df, df2, groups, subs, csdb

missing = issuers[~issuers['issuername'].notnull()]

issuers.to_csv("csv/issuers.csv", sep = ',', index=False)

# =============================================================================
# CHECK CSDB COVERAGE
# =============================================================================

#------------------------------------------------------------------------------
issuers = pd.read_csv("csv/issuers.csv", sep = ',', index_col = False)

bonds = issuers[issuers['issuername'].notnull()]
bonds.drop_duplicates(subset=['externalcode_isin'], keep = 'first', inplace = True) #SHIT TO BE FIXED IN CSDB
bonds['amountoutstanding_eur'].sum() # 975 Bn

csdb = extract_query('csdb_test')
csdb['amountoutstanding_eur'].sum() # 1.47 Tn

lei = bonds['guolei'].append(bonds['lei']).drop_duplicates()

mismatch = csdb[~csdb['issuerexternalcode_lei'].isin(lei)]
mismatch['amountoutstanding_eur'].sum() # 705 Bn

mismatch = mismatch.dropna(subset=['issuerexternalcode_lei'])
mismatch['amountoutstanding_eur'].sum() # 695 Bn

mismatch.to_csv("csv/mismatch.csv", sep = ',', index=False)

#------------------------------------------------------------------------------