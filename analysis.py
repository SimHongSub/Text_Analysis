import os
import sys
import pandas as pd  # 데이터 전처리
import numpy as np  # 데이터 전처리
import random  # 데이터 전처리
from pandas import DataFrame  # 데이터 전처리
from collections import Counter  # 데이터 전처리

from tqdm import tqdm  # 시간 측정용

from sklearn.feature_extraction.text import CountVectorizer  # model setting
from sklearn.model_selection import train_test_split  # model setting

from sklearn.naive_bayes import MultinomialNB  # model 관련
from sklearn.metrics import roc_auc_score  # model 성능 확인
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from eunjeon import Mecab  # SHS-형태소분석 라이브러리 추가
#import konlpy
# from konlpy.tag import Mecab
from datacreate import sampling_data

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
seed = random.randrange(sys.maxsize)

sampling_data(seed)

train_data = pd.read_csv(BASE_DIR+"/data/test/sample_train_data_"+str(seed)+".csv")
test_data = pd.read_csv(BASE_DIR+"/data/test/sample_test_data_"+str(seed)+".csv")

submission_data = pd.read_csv(BASE_DIR+"/data/submission_제출양식.csv")

random.seed(seed)

test_data['prediction'] = 2  # smishing 컬럼 추가

tokenizer = Mecab()

print("train data 토큰화")
train_doc = [(tokenizer.pos(x), y) for x, y in tqdm(zip(train_data['text'], train_data['smishing']))]

print("test data 토큰화")
test_doc = [(tokenizer.pos(x), y) for x, y in tqdm(zip(test_data['text'], test_data['smishing']))]

stopwords = ['XXX', '.', '을', '를', '이', '가', '-', '(', ')', ':', '!', '?', ')-', '.-', 'ㅡ', 'XXXXXX', '..', '.(', '은', '는']  # 필요없는 단어 리스트


def get_couple(_words):
    global stopwords
    _words = [x for x in _words if x[0] not in stopwords]
    length = len(_words)
    for i in range(length - 1):
        yield _words[i][0], _words[i + 1][0]


def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()


X_train = []  # text 값 리스트
Y_train = []  # 각 text에 대한 smishing 값 리스트

print("train data 토큰 필요없는 단어리스트 제거, 모형에 사용하기 위한 데이터 전처리")
for lwords in train_doc:
    Y_train.append(lwords[1])

    temp = []
    for x, y in get_couple(lwords[0]):
        temp.append("{}.{}".format(x, y))

    X_train.append(" ".join(temp))

X_test = []  # test데이터 text 값 리스트
Y_test = []

print("test data 토큰 필요없는 단어리스트 제거, 모형에 사용하기 위한 데이터 전처리")
for lwords in test_doc:
    Y_test.append(lwords[1])

    temp = []
    for x, y in get_couple(lwords[0]):
        temp.append("{}.{}".format(x, y))

    X_test.append(" ".join(temp))

v = CountVectorizer()

v.fit(X_train)

vec_x_train = v.transform(X_train).toarray()
vec_x_test = v.transform(X_test).toarray()

m1 = MultinomialNB()
m1.fit(vec_x_train, Y_train)

y_train_pred1 = m1.predict_proba(vec_x_train)
y_train_pred1_one = [i[1] for i in y_train_pred1]

y_test_pred1 = m1.predict_proba(vec_x_test)
y_test_pred1_one = [i[1] for i in y_test_pred1]

test_data['prediction'] = y_test_pred1_one

test_data.to_csv(BASE_DIR + "/data/result/sample_test_result_"+str(seed)+".csv", index=False)

auc = roc_auc_score(Y_test, y_test_pred1_one)
print("Seed : " + str(seed))
print('AUC: %.5f' % auc)

fpr, tpr, thresholds = roc_curve(Y_test, y_test_pred1_one)

plot_roc_curve(fpr, tpr)

#confusion_matrix(Y_test, y_test_pred1_one)

#print(classification_report(Y_test, y_test_pred1_one, target_names=['class 0', 'class 1']))

