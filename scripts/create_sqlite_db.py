# Create scripts/create_sqlite_db.py
import pandas as pd
import sqlite3

# Load processed data
df = pd.read_csv('data/processed/processed_telco_data.csv')

# Create SQLite database
conn = sqlite3.connect('data/telco.db')
df.to_sql('telco_customers', conn, if_exists='replace', index=False)
conn.close()

print("Database created successfully!")