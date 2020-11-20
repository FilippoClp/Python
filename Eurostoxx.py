import os as os                                                                # Connect to local filesystem
import pandas as pd
import numpy as np
import eikon as ek

# Eikon API key (search for 'APPKEY' in Eikon)
ek.set_app_key('510c1a92eb4c4b01802c252bb8195a0e11e88194')

# julius
ek.set_app_key('a3a9b6494ced47fabcbfab3641d79ff33fa4650c')
os.chdir("P:/ECB business areas/DGE/DMP/MAY/NFC vulnerability/EUROSTOXX/")

# =============================================================================
# CREATE DATABASE EUROSTOXX 50 STOCK PRICES
# =============================================================================

# Download the components of the EUTOSTOXX index and convert them to list
components = ek.get_data('0#.STOXX50E', 'TR.RIC')[0]['RIC']
components = components.tolist()
 
# Remove NOKIA because Eikon says that I have no permission to download it    
components.remove('NOKIA.HE')

# Get the time series of the EUROSTOXX index               
data = ek.get_timeseries('.STOXX50E',
                         ['CLOSE'],
                         start_date='2019-01-01',
                         interval='daily')

data = data.rename(columns={'CLOSE': '.STOXX50E'})

for component in components:
    df_temp = ek.get_timeseries([component],
                                   start_date='2019-01-01',
                                   interval='daily',
                                   fields='CLOSE')
    # Rename to prevent clash
    df_temp = df_temp.rename(columns={'CLOSE': component})

    # Join the two dataframes
    data = data.join(df_temp[component])

del component, components, df_temp

data.to_csv("Eurostoxx.csv", sep = ',', index=True)

# =============================================================================
# CREATE DATABASE STOXX600 BALANCE SHEET AND EARNINGS EXPECTATIONS
# =============================================================================

# Extract the components of the EUTOSTOXX index and convert them to list
isin = pd.read_csv("isin.csv", sep = ',', index_col = False)
isin = isin.ISIN.tolist()

# Extract RIC from ISIN
components = ek.get_symbology(isin, from_symbol_type='ISIN', to_symbol_type='RIC', bestMatch=True)
components = {'ISIN': components.index.tolist(), 'RIC': components.RIC.tolist()}
components = pd.DataFrame(components)
components = components.dropna()
del isin

# Remove RICs you have not access to 
for component in components.RIC:
    if str(component).endswith('HE'):
        components.drop(components.loc[components.RIC==component].index, inplace=True)

# Extract Components Stock Price
stock_prices = ek.get_timeseries('.STOXX', ['CLOSE'], start_date='2017-01-01', interval='quarterly')
stock_prices = stock_prices.rename(columns={'CLOSE': '.STOXX50E'})

for component in components.RIC:
    df_temp = ek.get_timeseries([component], start_date='2017-01-01', interval='quarterly', fields='CLOSE')
    
    # Rename to prevent clash
    df_temp = df_temp.rename(columns={'CLOSE': component})
    # Join the two dataframes
    stock_prices = stock_prices.join(df_temp[component])
    
stock_prices['date'] = stock_prices.index
stock_prices['date']= stock_prices['date'].dt.year.apply(str) + stock_prices['date'].dt.quarter.apply(str) 
stock_prices.to_csv("stock_prices.csv", sep = ',', index=True)
del component,  df_temp



# Upload and amend Balance sheet data
balance_sheet = pd.read_csv("thomson_update.csv", sep = ',', index_col = False)
balance_sheet['RIC'] = ''

for index, row in components.iterrows():
    balance_sheet.loc[balance_sheet['ISIN']==row['ISIN'], 'RIC'] = row['RIC']
del index, row


# Merge balance sheet with stock price
balance_sheet['stock_price'] = np.nan
stock_prices.index= stock_prices.date 

for column in stock_prices:
    for index, row in balance_sheet.loc[balance_sheet['RIC']==column].iterrows():
        x = stock_prices.loc[stock_prices['date']==row['date'], column]
        balance_sheet.loc[(balance_sheet['RIC']==row['RIC']) & (balance_sheet['date']==row['date']), 'stock_price'] = stock_prices.at[str(row['date']), column] 
        print(stock_prices[column])   
del column, index, row
    

# Upload earnings expectations FY1
ibes, err = ek.get_data(components.RIC.to_list(), ['TR.EPSMeanEstimate.value', 'TR.EPSMeanEstimate.origdate'], parameters={'SDate':'2017-03-10', 'EDate':'2020-06-30', 'Period':'CY1'})
ibes.drop_duplicates(inplace=True)   
del err

#convert ibes into quarterly
ibes['Activation Date'] = ibes['Activation Date'].str.slice(start=0, stop=10)
ibes['date'] = pd.to_datetime(ibes['Activation Date'], format='%Y-%m-%d', errors='ignore')
ibes['date']= ibes['date'].dt.year.apply(str) + ibes['date'].dt.quarter.apply(str) 

#reshape
ibes=ibes.groupby(['Instrument', 'date']).last().reset_index()
ibes=ibes.pivot(index='date', columns='Instrument', values='Earnings Per Share - Mean Estimate')

# Merge balance sheet with ibes
balance_sheet['IBES_CY1'] = np.nan
ibes['date'] = ibes.index

for column in ibes:
    for index, row in balance_sheet.loc[balance_sheet['RIC']==column].iterrows():
        balance_sheet.loc[(balance_sheet['RIC']==row['RIC']) & (balance_sheet['date']==row['date']), 'IBES_CY1'] = ibes.at[str(row['date']), column] 
        print(ibes[column])   
del column, index, row


# Upload earnings expectations FY0
ibes, err = ek.get_data(components.RIC.to_list(), ['TR.EPSMeanEstimate.value', 'TR.EPSMeanEstimate.origdate'], parameters={'SDate':'2017-03-10', 'EDate':'2020-06-30', 'Period':'CY0'})
ibes.drop_duplicates(inplace=True)   
del err

#convert ibes into quarterly
ibes['Activation Date'] = ibes['Activation Date'].str.slice(start=0, stop=10)
ibes['date'] = pd.to_datetime(ibes['Activation Date'], format='%Y-%m-%d', errors='ignore')
ibes['date']= ibes['date'].dt.year.apply(str) + ibes['date'].dt.quarter.apply(str) 

#reshape
ibes=ibes.groupby(['Instrument', 'date']).last().reset_index()
ibes=ibes.pivot(index='date', columns='Instrument', values='Earnings Per Share - Mean Estimate')

# Merge balance sheet with ibes
balance_sheet['IBES_CY0'] = np.nan
ibes['date'] = ibes.index

for column in ibes:
    for index, row in balance_sheet.loc[balance_sheet['RIC']==column].iterrows():
        balance_sheet.loc[(balance_sheet['RIC']==row['RIC']) & (balance_sheet['date']==row['date']), 'IBES_CY0'] = ibes.at[str(row['date']), column] 
        print(ibes[column])   
del column, index, row

#export
balance_sheet.to_csv("TR_balance_sheet.csv", sep = ',', index=False)


# =============================================================================
# CREATE DATABASE STOXX600 STOCK PRICES
# =============================================================================

# Extract the components of the EUTOSTOXX index and convert them to list
isin = pd.read_csv("isin.csv", sep = ',', index_col = False)
isin = isin.ISIN.tolist()

components = ek.get_symbology(isin, from_symbol_type='ISIN', to_symbol_type='RIC', bestMatch=True)
# components['RIC'].to_csv("RIC.csv", sep = ',', index=True)

# now also get the tkr and copy it to .csv for further use
ticker = ek.get_symbology(isin, from_symbol_type='ISIN', to_symbol_type='ticker', bestMatch=True)
ticker['ticker'].to_csv("ticker.csv", sep = ',', index=True)
del ticker

#aa=(components.RIC.tolist())
#indices = []
#for i in range(len(aa)):
#   if str(aa[i]) == 'nan':
#      indices.append(i)


components = [x for x in components.RIC.tolist() if str(x) != 'nan']


# index over the isins
i = 0
for component in components[:]:

    if component.endswith('HE'):
        components.remove(component)
 #       isin_store.remove(i)
 #   i = i + 1    

data = ek.get_timeseries('.STOXX',
                         ['CLOSE'],
                         start_date='2019-01-01',
                         interval='daily')

# giulio: modifico .STOXX che non piace al software
data = data.rename(columns={'CLOSE': 'STOXX'})

i = 0
for component in components:
    df_temp = ek.get_timeseries([component],
                                   start_date='2019-01-01',
                                   interval='daily',
                                   fields='CLOSE')


    # Rename to prevent clash
    retoisin = ek.get_symbology(component, from_symbol_type='RIC', to_symbol_type='ISIN', bestMatch=True)
#    jhser = retoisin.ISIN.tolist()
#    df_temp = df_temp.rename(columns={'CLOSE': jhser})
#    old
    df_temp = df_temp.rename(columns={'CLOSE': isin[i]})
    i = i + 1
#    df_temp = df_temp.rename(columns={'CLOSE': component})


    # Join the two dataframes
    data = data.join(df_temp[component])
  #  data = data.join(df_temp[jhser])

del component, components, df_temp, isin

data.to_csv("Stoxx6002.csv", sep = ',', index=True)

#this is to download the IBES 
ek.get_data(components, ['TR.EPSEstValue.value', 'TR.EPSEstValue.origdate', 'TR.EPSEstValue.broker_id'], parameters={'SDate':'2019-01-30', 'EDate':'2020-06-30', 'Period':'FY1'})

dati, err = ek.get_data(components.RIC.to_list(), ['TR.EPSMeanEstimate.value', 'TR.EPSMeanEstimate.origdate'], parameters={'SDate':'2019-09-30', 'EDate':'2020-06-30', 'interval': 'quarterly', 'Period':'CY0'})

dati.to_csv("IBES.csv", sep = ',', index=True)