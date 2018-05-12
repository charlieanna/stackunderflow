# Load pretrained Word2vec and FastText Vectors
import os
from gensim.models import FastText
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from flask import Flask, render_template
from flask import request
from google.cloud import bigquery
import operator
from nltk.util import ngrams
import matplotlib.pyplot as plt
import pickle
import csv #For importing data from a csv file
from collections import defaultdict
from sklearn.cluster import KMeans
        
app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/result',methods = ['POST', 'GET'])
def result():
   if request.method == 'POST':
      result = request.form
      word2vec_model, fasttext_model = load_models()
      # Once the model has been calculated, it is easy and fast to find most similar words.
      similar_words = fasttext_model.wv.most_similar([result['word']], topn=10)

      client = bigquery.Client.from_service_account_json('../../My_Project-c23185ac100b.json')
      # get all the questions on this tag selected by the user sorted descending, this way u can get the most important topics
  # or tags which are important.

      word = similar_words[0][0]
      query = """
              SELECT id, questions.tags as tags, questions.score as score, questions.title as title
              FROM `bigquery-public-data.stackoverflow.posts_questions` as questions
              where  questions.tags like @a
              order by questions.score desc
              limit 5
              """
      query_params = [
          bigquery.ScalarQueryParameter(
              'a', 'STRING', "%"+word+"%")
          ]
      job_config = bigquery.QueryJobConfig()
      job_config.query_parameters = query_params
      query_job = client.query(query, job_config=job_config)
      results = query_job.result()

      filename = 'finalized_model.sav'

      if not os.path.exists(filename):
        kmeans = KMeans(n_clusters=100 )
        X = word2vec_model[word2vec_model.wv.vocab]
        kmeans.fit(X)        
        pickle.dump(kmeans, open(filename, 'wb'))
      else:
        kmeans = pickle.load(open(filename, 'rb'))
      labels = kmeans.labels_

      word_cluster1 = {}
      for i, word in enumerate(word2vec_model.wv.vocab): 
          word_cluster1[word] = labels[i]

      v1 = defaultdict(list)
      v2 = {}
      for key, value in sorted(word_cluster1.items()):
          v1[value].append(key)
          v2[key] = value

      v = {}
      num = set()
      for word in similar_words:
        # show only matching clusters. 
        num.add(v2[word[0]])

      return render_template("result.html",result = result, similar_words = dict(similar_words), results = results, v1 = v1, v2 = v2, num=  num)

def load_data():
    file_name = "../data.csv"
    with open(file_name, 'r') as f:  #opens data file
        reader = csv.reader(f)
        data = list(list(rec) for rec in csv.reader(f, delimiter=',')) #reads csv into a list of lists
        return data

    

def load_models():
    #data = load_data()
    if not os.path.exists('../word2vec_model1000'):
        word2vec_model = Word2Vec(data, min_count=1000, size=200)
        word2vec_model.save("../word2vec_model1000")
    else:
        word2vec_model = KeyedVectors.load("../word2vec_model1000")
    if not os.path.exists('../fast_text_model1000'):
        fasttext_model = FastText(data, min_count=1000)
        fasttext_model.save("../fast_text_model1000")
    else:
        fasttext_model = FastText.load("../fast_text_model1000")
    return word2vec_model, fasttext_model

if __name__ == '__main__':
   app.run(debug = True)
