import pandas as pd
import os 

df = pd.read_excel('attendance_db.xlsx')
df.to_csv('attendance_data.csv', index=False)
os.remove('attendance_db.xlsx')

print("Excel file converted to CSV and original file removed.")