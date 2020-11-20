import os as os  
import pyodbc                                                                
import pandas as pd
import numpy as np
import eikon as ek

# Eikon API key (search for 'APPKEY' in Eikon)
ek.set_app_key('510c1a92eb4c4b01802c252bb8195a0e11e88194')

def extract_query(sql_filename):
    
    cnxn = pyodbc.connect("DSN=DISC DP Impala 64bit; UID=; PWD=", autocommit=True)
    cursor = cnxn.cursor()

    query = open("queries/" + sql_filename + ".sql", "r") 
    df = pd.read_sql_query(query.read(), cnxn)

    cursor.close() #Close connection
    cnxn.close() 
    
    return df

os.chdir("P:/ECB business areas/DGE/DMP/MAY/NFC vulnerability/Eikon NFCs/")

# =============================================================================
# CREATE DATABASE OF EURO AREA NFCS BALANCE SHEET
# =============================================================================

# Some firms will be delisted/Merged over time, that is why the match with the previous 

# Select all company RICs in the Euro Area
syntax = 'SCREEN(U(IN(Equity(active,public,primary))), IN(TR.RegCountryCode, DE, ES, FR, IT, NL, AT, BE, PT, FI, IE, LU, GR, SI, SK, LV, LT, EE, MT, CY), DOES_NOT_CONTAIN(TR.TRBCEconomicSector,Financials), CURN=EUR)'
names, e = ek.get_data(syntax, ['TR.CommonName', 'TR.RegCountryCode'])
del syntax, e

# Find ISIN codes and add them to the dataset
isin = ek.get_symbology(names.Instrument.to_list(), from_symbol_type='RIC', to_symbol_type='ISIN', bestMatch=True)
names['ISIN'] = isin['ISIN'].to_list()
del isin

###############################################################################

quarters = []
for i in range (2010,2021):
   for j in range (1,5):
       quarters.append(str(j) + 'CQ' + str(i))
del i, j     

quarters.remove('4CQ2020')

fields = ['TR.NetIncome.Date','TR.PropertyPlantEquipmentTotalNet','TR.NetIncome','TR.CapitalExpenditures','TR.NetSales','TR.NetDebt',
          'TR.EBITDA','TR.LTInvestments','TR.CashAndSTInvestments','TR.TotalAssetsReported', 'TR.PriceClose', 'TR.EPSMeanEstimate.value', 'TR.TRBCEconomicSector']

b_sheet = pd.DataFrame()

for q in quarters: 
    data, e = ek.get_data(names.Instrument.to_list(), fields, parameters={'Period': q})
    data['Date'] = q
    b_sheet = b_sheet.append(data, ignore_index=True)
del fields, e, q, quarters



b_sheet['Date'] = b_sheet['Date'].str.replace('CQ', ' ')
b_sheet['Date'] = 'Q' + b_sheet['Date']
b_sheet['Date'] = pd.to_datetime(b_sheet['Date'].str.replace(r'(Q\d) (\d+)', r'\2-\1'), errors='ignore')
b_sheet.sort_values(by=['Instrument', 'Date'], inplace=True)


###################################################################################

# Retrieve balance sheet variables
#fields = ['TR.NetIncome.Date','TR.PropertyPlantEquipmentTotalNet','TR.NetIncome','TR.CapitalExpenditures','TR.NetSales','TR.NetDebt',
#          'TR.EBITDA','TR.LTInvestments','TR.CashAndSTInvestments','TR.TotalAssetsReported', 'TR.PriceClose', 'TR.EPSMeanEstimate.value', 'TR.TRBCEconomicSector']
#data, e = ek.get_data(names.Instrument.to_list(), fields, parameters={'Frq':'Q', 'Period':'FQ0', 'SDate':'2010-01-01', 'EDate':'2020-09-30'})
#del fields, e

# Select all company RICs in the Euro Area
#syntax = 'SCREEN(U(IN(Equity(active,public,primary))), IN(TR.RegCountryCode, NL, AT, BE, PT, FI, IE, LU, GR, SI, SK, LV, LT, EE, MT, CY), DOES_NOT_CONTAIN(TR.TRBCEconomicSector,Financials), CURN=EUR)'
#names2, e = ek.get_data(syntax, ['TR.CommonName', 'TR.RegCountryCode'])
#del syntax, e

# Find ISIN codes and add them to the dataset
#isin = ek.get_symbology(names2.Instrument.to_list(), from_symbol_type='RIC', to_symbol_type='ISIN', bestMatch=True)
#names2['ISIN'] = isin['ISIN'].to_list()
#del isin

# Retrieve balance sheet variables
#fields = ['TR.NetIncome.Date','TR.PropertyPlantEquipmentTotalNet','TR.NetIncome','TR.CapitalExpenditures','TR.NetSales','TR.NetDebt',
#          'TR.EBITDA','TR.LTInvestments','TR.CashAndSTInvestments','TR.TotalAssetsReported', 'TR.PriceClose', 'TR.EPSMeanEstimate.value', 'TR.TRBCEconomicSector']
#data2, e = ek.get_data(names2.Instrument.to_list(), fields, parameters={'Frq':'Q', 'Period':'FQ0','SDate':'2010-01-01', 'EDate':'2020-09-30'})
#del fields, e

#names = names.append(names2, ignore_index=True)
#names.to_csv("components.csv", sep = ',', index=False)

## GIULIO
#aa = data.merge(names, on='Instrument')
#bb = data2.merge(names2, on='Instrument')
#data_eikon=aa.append(bb)
#use non merged dataset 
#data_eikon.to_csv("only_eikon.csv", sep = ',', index=False)

#data = data.append(data2, ignore_index=True)
#b_sheet = names.merge(data, on='Instrument', how='left')
#b_sheet.to_csv("raw_b_sheet.csv", sep = ',', index=False)
#del data2
#del names2

bs = names.merge(b_sheet, on='Instrument', how='left')
bs.to_csv("raw_b_sheet.csv", sep = ',', index=False)

# =============================================================================
# OPEN PRE-PROCESSED EIKON DATABASE AND PRODUCE BALANCED PANEL                                                    
# =============================================================================

df = pd.read_csv("raw_b_sheet.csv", sep = ',', index_col = False)

df.dropna(how='all', subset=['Date'],inplace=True)
df.dropna(how='all', subset=['Property/Plant/Equipment, Total - Net','Net Income Incl Extra Before Distributions','Capital Expenditures, Cumulative','Net Sales','EBITDA'],inplace=True)
df.drop_duplicates(inplace=True)

df['Date'] = df['Date'].str.slice(start=0, stop=10)
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d', errors='ignore')
df['Date'] = df['Date'].dt.to_period('Q')


# =============================================================================
# OPEN PRE-PROCESSED EIKON DATABASE AND PRODUCE BALANCED PANEL                                                    
# =============================================================================

# Extract Identifier from ORBIS and add LEI to the dataset
orbis = extract_query("orbis")

df = df.merge(orbis, left_on='ISIN', right_on='isinnumber', how='left')
df.drop(columns=['isinnumber'], inplace=True)
del orbis

# Harmonize the CSDB identifiers 
csdb = extract_query("csdb")

for name in csdb["issuername"].drop_duplicates():
    for code in csdb.loc[csdb['issuername'] == name, 'id']:
        if len(code) == 20:
            csdb.loc[csdb['issuername'] == name, 'id'] = code
            break
del code, name

# Format date to prepare the Merge
csdb['Date'] = pd.to_datetime(csdb['quarter'], format='%Y-%m-%d', errors='ignore')
csdb['Date'] = csdb['Date'].dt.to_period('Q')    
csdb.drop(columns=['quarter', 'issuername'], inplace=True)

# Merge datasets on LEI and BVD
df = df.merge(csdb, left_on=['lei','Date'], right_on=['id','Date'], how='left')
df = df.merge(csdb, left_on=['bvd','Date'], right_on=['id','Date'], how='left')

df['Debt Securities'] = df['sum(amountoutstanding_eur)_x'].fillna(0) + df['sum(amountoutstanding_eur)_y'].fillna(0)
df['Debt Securities'].replace(0, np.nan, inplace=True)
df.drop(columns=['sum(amountoutstanding_eur)_x', 'sum(amountoutstanding_eur)_y', 'id_x', 'id_y'], inplace=True)

df.rename(columns={'Instrument':'RIC', 'Company Common Name':'Name', 'Country ISO Code of Incorporation':'Country', 'bvd':'BVD', 'lei':'LEI'}, inplace=True)

df.to_csv("merged_new_3.csv", sep = ',', index=False)   

df2 = pd.read_csv("merged_new_2.csv", sep = ',', index_col = False)
 





