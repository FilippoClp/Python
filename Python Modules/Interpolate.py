import os as os                      # Connect to local filesystem
import pandas as pd                  # Import data in a pandas dataframe 

os.chdir("P:\\ECB business areas\\DGE\\DMP\\MAY\\NFC vulnerability\\Python_Project")

df = pd.read_csv("csv/interpol.csv", sep = ',', index_col = False)

df['date'] = pd.to_datetime(df['date'])

df.index = df['date']

df['min'] = df['min'].interpolate(method = 'cubic', limit_direction = 'both')
df['med'] = df['med'].interpolate(method = 'cubic')
df['max'] = df['max'].interpolate(method = 'cubic')

