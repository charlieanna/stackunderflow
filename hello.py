# Load pretrained Word2vec and FastText Vectors
import os
from gensim.models import FastText
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from flask import Flask, render_template
from flask import request
import csv #For importing data from a csv file
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
      return render_template("result.html",result = result, similar_words = dict(similar_words))

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
