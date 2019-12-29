# 2019.12.28 분석 코드

import os
import time

import numpy as np
import pandas as pd
import random

from konlpy.tag import Kkma
from konlpy.utils import pprint

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score

from collections import Counter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
start = time.time()

# data road
print("1. train data road")
train = pd.read_csv("test.csv")
print("2. test data road")
test = pd.read_csv(BASE_DIR+"/data/train.csv")

random.seed(2019)
test_list = list(test['smishing'].index)
test_sample_index = random.sample(test_list, 1600)

test = test.iloc[test_sample_index, :].reset_index(drop=True)

print(np.array(test['smishing']))
# kkma tokenizer
kkma = Kkma()

# stopword define
stopwords = ['XXX', '.', '을', '를', '이', '가', '-', '(', ')', ':', '!', '?', ')-', '.-', 'ㅡ', 'XXXXXX', '..', '.(', '은', '는']

# tokenization
print("3. train data tokenization")
tokens = []
for data in train['text']:
    tokens.append(kkma.morphs(data))

print("4. test data tokenization")
test_tokens = []
for data in test['text']:
    test_tokens.append(kkma.morphs(data))

# cleaning
print("5. train data cleaning")
refined_token = []
for words in tokens:
    token = []
    for word in words:
        if word not in stopwords:
            token.append(word)

    refined_token.append(token)

print("6. test data cleaning")
test_refined_token = []
for words in test_tokens:
    token = []
    for word in words:
        if word not in stopwords:
            token.append(word)

    test_refined_token.append(token)

print("7. train data word sum")
words = sum(refined_token, [])

print("8. test data word sum")
test_words = sum(test_refined_token, [])

# Integer encoding
#words = sum(refined_token, [])

# 빈도수 체크
# vocab = Counter(words)
# print("vocab : ")
# print(vocab)

# 빈도수가 낮은 단어 제거하는 작업 추가하면 다시 한번 전처리 가능
# vocab = vocab.most_common()

# 빈도수에 따른 인덱싱 작업
# word_to_index = {}
# i = 0
# for (word, frequency) in vocab:
#     i += 1
#     word_to_index[word] = i
#
# print("word_to_index : ")
# print(word_to_index)

# 학습시키지 못한 유의미한 글자 데이터를 만들 수 있을까? : BPE 알고리즘

dtmVector = CountVectorizer()

print("9. train data tfidf")
X_train_dtm = dtmVector.fit_transform(words)
tfidf_transformer = TfidfTransformer()
tfidfv = tfidf_transformer.fit_transform(X_train_dtm)

mod = MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
mod.fit(tfidfv, np.array(train['smishing']))

X_test_dtm = dtmVector.transform(test_words)
tfidfv_test = tfidf_transformer.transform(X_test_dtm)

predicted = mod.predict(tfidfv_test)
print("정확도 : ", accuracy_score(np.array(test['smishing']), predicted))

print("time : ", time.time() - start)
