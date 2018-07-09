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
from flask_mail import Mail
from flask_mail import Message
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.jobstores.mongodb import MongoDBJobStore
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.executors.pool import ThreadPoolExecutor, ProcessPoolExecutor
import time
import atexit
# from config import config
import requests
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from flask_api import FlaskAPI

app = FlaskAPI(__name__)

# app.config['MAIL_SERVER']=config['MAIL_SERVER']
# app.config['MAIL_PORT'] = config['MAIL_PORT']
# app.config['MAIL_USERNAME'] = config['MAIL_USERNAME']
# app.config['MAIL_PASSWORD'] = config['MAIL_PASSWORD']
# app.config['MAIL_USE_TLS'] = config['MAIL_USE_TLS']
# app.config['MAIL_USE_SSL'] = config['MAIL_USE_SSL']

mail = Mail(app)


@app.route("/")
def index():
    # defining the api-endpoint
    # API_ENDPOINT = "https://stackoverflow.com/oauth/access_token"
    # # data to be sent to api
    # data = {'client_id':12430,
    #         'client_secret':'9*Kyrtrtb*iwc6v4soDAuw((',
    #         'code':'DNkk4aD9QkkCjzvvRQc8uQ))',
    #         'redirect_uri':'http://localhost:5000'}

    # # sending post request and saving response as response object
    # r = requests.post(url = API_ENDPOINT, data = data)
    # print(r)
    # # extracting response text
    # response = r.text
    # access_token = response.split("=")[1]
    # print("access_token:", response.split("=")[1])


    jobstores = {
        'default': SQLAlchemyJobStore(url='sqlite:///jobs.sqlite')
    }
    executors = {
        'default': ThreadPoolExecutor(20),
        'processpool': ProcessPoolExecutor(5)
    }
    job_defaults = {
        'coalesce': False,
        'max_instances': 3
    }
    scheduler = BackgroundScheduler(jobstores=jobstores, executors=executors, job_defaults=job_defaults)

    # scheduler.add_job(
    # func=send_questions,
    # trigger=IntervalTrigger(days = 1),
    # replace_existing=True)
    from datetime import date
    from datetime import datetime
    from datetime import timedelta
    scheduler.add_job(send_questions, 'date', run_date=datetime.now() + timedelta(seconds=2)  ,replace_existing=True)

    # scheduler.add_job(print_date_time, 'date', run_date=date(2009, 11, 6), args=['text'])
    # scheduler.start()
    # Shut down the scheduler when exiting the app
    atexit.register(lambda: scheduler.shutdown())

    return render_template('index.html')

@app.route("/top")
def top_questions():
  import json
  from pprint import pprint
  data = None
  with open('../tag_count_score_hash.json') as f:
    data = json.load(f)

  response = {'data': data}
  return response

@app.route("/question")
def question():
  # get all the anwswers of this question
  client = bigquery.Client.from_service_account_json('../../My_Project-c23185ac100b.json')
  # get all the questions on this tag selected by the user sorted descending, this way u can get the most important topics
  # or tags which are important.
  # get all the questions on this tag selected by the user sorted descending, this way u can get the most important topics
# or tags which are important.
  result = request.args
  question_id = result.get("id")
  query = """
          SELECT answers.id as id, answers.score as ascore, questions.title as question
          FROM `bigquery-public-data.stackoverflow.posts_questions` as questions
          INNER JOIN `bigquery-public-data.stackoverflow.posts_answers` as answers
          on questions.id = answers.parent_id
          where  questions.id = @a
          order by answers.score desc limit 1
          """
  query_params = [
      bigquery.ScalarQueryParameter(
          'a', 'INT64', question_id)
      ]
  job_config = bigquery.QueryJobConfig()
  job_config.query_parameters = query_params
  query_job = client.query(query, job_config=job_config)
  results = query_job.result()
  return {'results': [{'id': row.id,'ascore': row.ascore, 'question': row.question} for row in results]}

def send_questions():
  with app.app_context():
    username = "Ankur Kothari"
    msg = Message("New questions for the day",
      sender="ankothari@gmail.com",
      recipients=["akotha01@syr.edu"])
    word2vec_model, fasttext_model = load_models()
      # Once the model has been calculated, it is easy and fast to find most similar words.
    similar_words = fasttext_model.wv.most_similar(['algorithms'], topn=20)
    client = bigquery.Client.from_service_account_json('../../My_Project-c23185ac100b.json')
    word = similar_words[0][0]
    query = """
            SELECT id, questions.tags as tags, questions.score as score, questions.title as title
            FROM `bigquery-public-data.stackoverflow.posts_questions` as questions
            where  questions.tags like @a
            and title not like "%closed%"
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

    msg.body = 'Hello '+username+',\nThese are the new questions for you, can you answer them?'
    msg.html = render_template('/mails/mail.html', username=username, results = results, similar_words = dict(similar_words))

    mail.send(msg)
    print("time", time.strftime("%A, %d. %B %Y %I:%M:%S %p"))
@app.route('/similar',methods = ['POST', 'GET'])
def similar():
  result = request.args
  question_id = result.get("id")
  query = """
          SELECT answers.id as id, answers.score as ascore, questions.title as question
          FROM `bigquery-public-data.stackoverflow.posts_questions` as questions
          INNER JOIN `bigquery-public-data.stackoverflow.posts_answers` as answers
          on questions.id = answers.parent_id
          where  questions.id = @a
          order by answers.score desc limit 1
          """
  query_params = [
      bigquery.ScalarQueryParameter(
          'a', 'INT64', question_id)
      ]
  job_config = bigquery.QueryJobConfig()
  job_config.query_parameters = query_params
  query_job = client.query(query, job_config=job_config)
  results = query_job.result()
  return {'results': [{'id': row.id,'ascore': row.ascore, 'question': row.question} for row in results]}

@app.route('/result',methods = ['POST', 'GET'])
def result():
  all_questions = []
  if request.method == 'POST':
    result = request.form
    import json
    from pprint import pprint
    tag_count_hash = None
    with open('../tag_count_score_hash.json') as f:
      tag_count_hash = json.load(f)
      word2vec_model = load_models()
      tags_similar = word2vec_model.wv.most_similar(result['word'], topn=30)
      related_tags = []
      for tag, similarity in tags_similar:
          related_tags.append({tag:tag_count_hash[tag]})

    response = {'data': tag_count_hash[result['word']], 'related_tags': related_tags}
    return response
    # result = request.form
    # word2vec_model, fasttext_model = load_models()
    # # Once the model has been calculated, it is easy and fast to find most similar words.
    # similar_words = fasttext_model.wv.most_similar([result['word']], topn=10)
    # client = bigquery.Client.from_service_account_json('../../My_Project-c23185ac100b.json')
    # questions = {}
    # for similar_word in similar_words:
    #   word = similar_word[0]
    #   query = """
    #           SELECT id, questions.tags as tags, questions.score as score, questions.title as title
    #           FROM `bigquery-public-data.stackoverflow.posts_questions` as questions
    #           where  questions.tags like @a
    #           and title not like "%closed%"
    #           order by questions.score desc
    #           limit 20
    #           """
    #   query_params = [
    #       bigquery.ScalarQueryParameter(
    #           'a', 'STRING', "%"+word+"%")
    #       ]
    #   job_config = bigquery.QueryJobConfig()
    #   job_config.query_parameters = query_params
    #   query_job = client.query(query, job_config=job_config)
    #   results = query_job.result()
    #   questions[word] = [[row.tags, row.id,row.score, row.title] for row in results]
    #   all_questions += questions[word]
    # filename = 'finalized_model.sav'

    # if not os.path.exists(filename):
    #   kmeans = KMeans(n_clusters=100 )
    #   X = word2vec_model[word2vec_model.wv.vocab]
    #   kmeans.fit(X)
    #   pickle.dump(kmeans, open(filename, 'wb'))
    # else:
    #   kmeans = pickle.load(open(filename, 'rb'))
    # labels = kmeans.labels_

    # word_cluster1 = {}
    # for i, word in enumerate(word2vec_model.wv.vocab):
    #     word_cluster1[word] = labels[i]

    # v1 = defaultdict(list)
    # v2 = {}
    # for key, value in sorted(word_cluster1.items()):
    #     v1[int(value)].append(key)
    #     v2[key] = int(value)

    # v = {}
    # num = set()
    # for word in similar_words:
    #   # show only matching clusters.
    #   num.add(v2[word[0]])
    # pending_questions = {}
    # # for similar_word in similar_words:
    # #   word = similar_word[0]
    # #   query = """
    # #           SELECT id, questions.tags as tags, questions.score as score, questions.title as title,  questions.body as body
    # #           FROM `bigquery-public-data.stackoverflow.posts_questions` as questions
    # #           where  questions.tags like @a and questions.score > 0
    # #           order by questions.score
    # #           limit 20
    # #           """
    # #   word = similar_words[0][0]
    # #   query_params = [
    # #       bigquery.ScalarQueryParameter(
    # #           'a', 'STRING', "%"+word+"%")
    # #       ]
    # #   job_config = bigquery.QueryJobConfig()
    # #   job_config.query_parameters = query_params
    # #   query_job = client.query(query, job_config=job_config)
    # #   results1 = query_job.result()
    # #   pending_questions[word] = [[row.tags, row.id,row.score, row.title, row.body] for row in results1]

    # return {'all_questions': all_questions, 'similar_words': dict(similar_words), 'questions': questions, 'v2': dict(v2), 'v1': v1, 'num': list(num)}


def load_data():
    file_name = "../data.csv"
    with open(file_name, 'r') as f:  #opens data file
        reader = csv.reader(f)
        data = list(list(rec) for rec in csv.reader(f, delimiter=',')) #reads csv into a list of lists
        return data




def load_models():
    data = load_data()
    if not os.path.exists('../word2vec_model'):
        word2vec_model = Word2Vec(data, min_count=1000, size=200)
        word2vec_model.save("../word2vec_model")
    else:
        word2vec_model = KeyedVectors.load("../word2vec_model")
    # if not os.path.exists('../fast_text_model1000'):
    #     fasttext_model = FastText(data, min_count=1000)
    #     fasttext_model.save("../fast_text_model1000")
    # else:
    #     fasttext_model = FastText.load("../fast_text_model1000")
    return word2vec_model#, fasttext_model

if __name__ == '__main__':
   app.run(debug = True)
