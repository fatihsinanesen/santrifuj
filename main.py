import os
from werkzeug.utils import secure_filename
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from TurkishStemmer import TurkishStemmer
import pandas as pd
import numpy as np
from io import BytesIO
import xlsxwriter
import openpyxl
import re
import nltk
from flask import Flask, send_file, render_template, request

nltk.download('stopwords')
stemmer = TurkishStemmer()
current_path = os.getcwd() + "\\"
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    try:
        if request.method == 'POST':
            #Önce gelen dosyayı oku (index.html'deki ayardan ötürü sadece xlsx gelebilir)
            file = request.files['file']
            filename = secure_filename(file.filename)
            selected_model = pickle.load(open(current_path + "ai_model.pkl", 'rb'))
            cv = pickle.load(open(current_path + "ai_vectorizer.pkl", 'rb'))
            test_data = pd.read_excel(file)

            #Sonra gelen dosyayı işle
            sonuc_df = sonuc_hazirla(test_data, cv, selected_model)
            
            #Sonra da yeni dosyayı hazırla
            output = BytesIO()
            writer = pd.ExcelWriter(output,engine='xlsxwriter')
            sonuc_df.to_excel(writer)
            writer.save()
            output.seek(0)
            return send_file(output, download_name="sonuc_" + filename)
    except Exception as ex:
        #Sıkıntı olursa başa dön
        return render_template("index.html")

    return

#Veriyi temizle ve hazırla
def prepare_data(data,text_column=0):
    corpus = []
    i = 0
    for i in range(0, len(data)):
        try:
            x = data.iloc[i][text_column]
            sentence = re.sub("\W+", " ", x)
            sentence = sentence.replace("İ","i")
            sentence = sentence.replace("I","ı")
            sentence = sentence.lower()
            sentence = sentence.replace("projesi","")
            sentence = sentence.replace("ar-ge","")
            sentence = sentence.replace("arge","")
            sentence = sentence.replace("teknoloji","")
            sentence = sentence.replace("araştırılması","")
            sentence = sentence.replace("geliştirilmesi","")
            word_list = sentence.split()
            stemmed_word_list = []
            
            for word in word_list:
                if not word in stopwords.words('turkish'):
                    if word.isalpha():
                        try:
                            stemmed_word_list.append(stemmer.stem(word))
                        except Exception as ex2:
                            pass

            new_sentence = re.sub("\W+"," ",re.sub(r'\b[a-zA-Z]{1}\b', " ", re.sub("\d", "", " ".join(stemmed_word_list))))
            corpus.append(new_sentence)
        except Exception  as ex:
            x = data.iloc[i][text_column]
            corpus.append(" ")
            print("err docid:"+str(i)+":"+str(x)+"---")
            print(ex)
    return(corpus)

#Tahmin sonuclarını hazırla
def sonuc_hazirla(test_data, cv, model):
    test_data_corpus = prepare_data(test_data, text_column=0)
    cv_test = CountVectorizer(max_features=10000,vocabulary=cv.get_feature_names_out())
    X_test_gercek = cv_test.fit_transform(test_data_corpus)
    ongoruler = pd.DataFrame(model.predict(X_test_gercek))

    yuzde = pd.DataFrame(model.predict_proba(X_test_gercek))
    yuzde.columns = model.classes_
    yuzdeT=yuzde.T

    ongoru1, ongoru2, ongoru3 = [], [], []

    for i in range(0, len(yuzde)):
      yuzdeT = yuzdeT.sort_values(by=[i], ascending=False)
      ongoru1.append(yuzdeT.index[0])
      ongoru2.append(yuzdeT.index[1])
      ongoru3.append(yuzdeT.index[2])

    list_tuples = list(zip(ongoru1, ongoru2, ongoru3))  
    dframe = pd.DataFrame(list_tuples, columns=['Alan 1', 'Alan 2', 'Alan 3'])
    rapor = pd.concat([test_data, dframe], axis=1)

    return(rapor)

if __name__ == "__main__":
    app.run()