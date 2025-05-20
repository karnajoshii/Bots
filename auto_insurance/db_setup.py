import sqlite3
import pandas as pd

# Load CSV data
csv_file = "data/auto_insurance.csv"
df = pd.read_csv(csv_file)

# Connect to SQLite database
conn = sqlite3.connect("insurance.db")
cursor = conn.cursor()

# Create table based on CSV structure
cursor.execute("""
CREATE TABLE IF NOT EXISTS auto_insurance (
    policy_id INTEGER PRIMARY KEY,
    customer_name TEXT,
    vehicle_model TEXT,
    premium_amount REAL,
    claim_status TEXT
);
""")

# Insert data
df.to_sql("auto_insurance", conn, if_exists="replace", index=False)

print("Database setup complete.")
conn.commit()
conn.close()
