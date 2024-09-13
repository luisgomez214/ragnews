# sqlalchemy <- this library "database independent"
import sqlite3
db = sqlite3.connect('ragnews.db')
# no standard file extension for sqlite;
# you can see: .db, .sql, .sqlite, .sqlite3

cursor = db.cursor()
sql = '''
SELECT count(*) FROM articles;
'''
cursor.execute(sql)
row = cursor.fetchone()
print(f"row={row}")

cursor = db.cursor()
sql = '''
SELECT title FROM articles('trump Harris debate')
'''
cursor.execute(sql)
rows = cursor.fetchall()
for row in rows:
    print(f"row={row}")
# make it ordered by how well the result matches the query
# take the top 10 of those
