import pandas as pd
import os 

df = pd.read_excel('employees.xlsx')
df.to_csv('employees.csv', index=False)
os.remove('employees.xlsx')

print("Excel file converted to CSV and original file removed.")