import psycopg2
import sys, os

## ****** LOAD PSQL DATABASE ***** ##


# Set up a connection to the postgres server.
# conn_string = "host="+ "stats@localhost" +" port="+ "5432" +" dbname="+ "stats" +" user=" + "charlieana" \
# +" password="+ "akk322"

 # psycopg2.connect(database='testdb', user='postgres',
 #        password='s$cret')
con=psycopg2.connect(database='stackoverflow', user='charlieanna',
        password='akk322')
print("Connected!")




with con:

    cur = con.cursor()
    cur.execute("select * from posts where tags like '%<go>%' order by score;")

    rows = cur.fetchall()

    for row in rows:
        print(row)

# create another table so that matching tags to questions become easier.

# fr each question extract the tags and then map the tags to question.s but the table will be huge because of the query. 
