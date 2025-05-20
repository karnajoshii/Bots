import sqlite3
import pandas as pd

# Path to your SQLite database
db_path = 'insurance.db'

# Connect to the SQLite database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# SQL query to get customers with the highest claim amounts
query ="""
SELECT strftime('%m', "Effective To Date") AS "Month", COUNT(*) AS "Number of Claims"
FROM insurance
WHERE "State" = 'California' AND "Months Since Last Claim" <= 6
GROUP BY strftime('%m', "Effective To Date")
ORDER BY strftime('%m', "Effective To Date");
"""
# month wise no of claims from California for last 6 months
# How much percentage of claims by males with different education level from Nevada



# """
# SELECT "Education",
#        (COUNT(*) * 100.0 / (SELECT COUNT(*) FROM insurance WHERE "Gender" = 'M' AND "State" = 'Arizona')) AS "Percentage of Claims"
# FROM insurance
# WHERE "Gender" = 'M' AND "State" = 'Arizona'
# GROUP BY "Education";
# """

# query = """
# SELECT "State", strftime('%Y', "Effective To Date") AS "Year", COUNT(*) AS "Total Claims"
# FROM insurance
# WHERE "State" = 'Arizona'
# GROUP BY "State", strftime('%Y', "Effective To Date");
# """


# SELECT "Customer", MAX("Total Claim Amount") AS "Highest Claim Amount"
# FROM "insurance"
# GROUP BY "Customer";

# Execute the query and load the result into a DataFrame
df = pd.read_sql_query(query, conn)

# Print the result
print("Top 10 customers with the highest claim amounts:")
print(df)

# Close the connection
conn.close()