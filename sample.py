import os
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
from eunjeon import Mecab  # SHS-형태소분석 라이브러리 추가
#import konlpy
# from konlpy.tag import Mecab

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

train_data = pd.read_csv(BASE_DIR+"/data/train.csv")    # 스미싱 데이터(1) : 18703개, 스미싱 데이터 X (0) : 277242개
test_data = pd.read_csv(BASE_DIR+"/data/public_test.csv")
submission_data = pd.read_csv(BASE_DIR+"/data/submission_제출양식.csv")

random.seed(2019)
train_nsm_list = list(train_data[train_data['smishing'] != 1].index)
train_nsmishing = random.sample(train_nsm_list, 18703)    # under sampling
train_smishing = list(train_data[train_data['smishing'] == 1].index)

sampling_train_data = train_data.iloc[train_smishing + train_nsmishing, :].reset_index(drop=True)

test_data['smishing'] = 2  # smishing 컬럼 추가

tokenizer = Mecab()

print("train data 토큰화")
train_doc = [(tokenizer.pos(x), y) for x, y in tqdm(zip(sampling_train_data['text'], sampling_train_data['smishing']))]

print("test data 토큰화")
test_doc = [(tokenizer.pos(x), y) for x, y in tqdm(zip(test_data['text'], test_data['smishing']))]

stopwords = ['XXX', '.', '을', '를', '이', '가', '-', '(', ')', ':', '!', '?', ')-', '.-', 'ㅡ', 'XXXXXX', '..', '.(', '은', '는']  # 필요없는 단어 리스트


def get_couple(_words):
    global stopwords
    _words = [x for x in _words if x[0] not in stopwords]
    length = len(_words)
    for i in range(length - 1):
        # if i == 0:
        #     print("get couple")
        #     print(_words[i][0])
        #     print(_words[i + 1][0])
        yield _words[i][0], _words[i + 1][0]


X_train = []  # text 값 리스트
Y_train = []  # 각 text에 대한 smishing 값 리스트

print("train data 토큰 필요없는 단어리스트 제거, 모형에 사용하기 위한 데이터 전처리")
j = 0
for lwords in train_doc:
    if j == 0:
        print(lwords)
        print(lwords[0])
        print(lwords[1])
        j += 1

    Y_train.append(lwords[1])

    temp = []
    for x, y in get_couple(lwords[0]):
        temp.append("{}.{}".format(x, y))
        if j == 1:
            print("temp : ")
            print(temp)
            j += 1

    X_train.append(" ".join(temp))
    if j == 2:
        print(temp)
        print(" ".join(temp))
        print("X_train : ")
        print(X_train)
        j += 1

X_test = []  # test데이터 text 값 리스트

print("test data 토큰 필요없는 단어리스트 제거, 모형에 사용하기 위한 데이터 전처리")
for lwords in test_doc:

    temp = []
    for x, y in get_couple(lwords[0]):
        temp.append("{}.{}".format(x, y))

    X_test.append(" ".join(temp))

v = CountVectorizer()

v.fit(X_train)
print("countvectorizer : ")
print(v.vocabulary_)

vec_x_train = v.transform(X_train).toarray()
print("vec_x_train : ")
print(vec_x_train[0])
vec_x_test = v.transform(X_test).toarray()

m1 = MultinomialNB()
m1.fit(vec_x_train, Y_train)

y_train_pred1 = m1.predict_proba(vec_x_train)
y_train_pred1_one = [i[1] for i in y_train_pred1]

y_test_pred1 = m1.predict_proba(vec_x_test)
y_test_pred1_one = [i[1] for i in y_test_pred1]

print("train data 스미싱 확률")
print(y_train_pred1)
print(y_train_pred1_one)

sampling_train_data['smishing'] = y_train_pred1_one

sampling_train_data.to_csv("sample_train.csv", index=False)

print("test data 스미싱 확률")
print(y_test_pred1)
print(y_test_pred1_one)

submission_data['smishing'] = y_test_pred1_one

submission_data.to_csv("sample_submission.csv", index=False)

#auc = roc_auc_score()

