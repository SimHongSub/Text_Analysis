# 2019.12.29 분석 코드 - AUC : 0.711732118

# -*- coding: utf-8 -*-

import os
import pandas as pd  # 데이터 전처리
import random  # 데이터 전처리

from tqdm import tqdm  # 시간 측정용

from sklearn.feature_extraction.text import CountVectorizer  # model setting

from sklearn.naive_bayes import MultinomialNB  # model 관련
from eunjeon import Mecab  # SHS-형태소분석 라이브러리 추가

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# train_data = pd.read_csv(BASE_DIR+"/data/train.csv")    # 스미싱 데이터(1) : 18703개, 스미싱 데이터 X (0) : 277242개
train_data = pd.read_csv(BASE_DIR+"/data/modified_data.csv")
test_data = pd.read_csv(BASE_DIR+"/data/public_test.csv")
submission_data = pd.read_csv(BASE_DIR+"/data/submission_제출양식.csv")

random.seed(2019)

train_list = list(train_data['smishing'].index)   # train 데이터 모든 index 리스트
selected_list = []  # train 데이터에서 추출된 일부 index 리스트 - 37403개
# stopwords = ['XXX', '.', '을', '를', '이', '가', '-', '(', ')', ':', '!', '?', ')-', '.-', 'ㅡ', 'XXXXXX', '..', '.(', '은', '는']  # 필요없는 단어 리스트
stopwords = ['XXX', '.', '을', '를', '이', '가', '-', '(', ')', ':', '!', '?', ')-', '.-', 'ㅡ', 'XXXXXX', '..', '.(', '은', '는', '어찌됐든','그위에','게다가','점에서 보아','비추어 보아','고려하면','하게될것이다','일것이다','비교적','좀','보다더','비하면','시키다','하게하다','할만하다','의해서','연이서','이어서','잇따라','뒤따라','뒤이어','결국','의지하여','기대여','통하여','자마자','더욱더','불구하고','얼마든지','마음대로','주저하지 않고','곧','즉시','바로','당장','하자마자','밖에 안된다','하면된다','그래','그렇지','요컨대','다시 말하자면','바꿔 말하면','즉','구체적으로','말하자면','시작하여','시초에','이상','허','헉','허걱','바와같이','해도좋다','해도된다','게다가','더구나','하물며','와르르','팍','퍽','펄렁','동안','이래','하고있었다','이었다','에서','로부터','까지','예하면','했어요','해요','함께','같이','더불어','마저','마저도','양자','모두','습니다','가까스로','하려고하다','즈음하여','다른','다른 방면으로','해봐요','습니까','했어요','말할것도 없고','무릎쓰고','개의치않고','하는것만 못하다','하는것이 낫다','매','매번','들','모','어느것','어느','로써','갖고말하자면','어디','어느쪽','어느것','어느해','어느 년도','라 해도','언젠가','어떤것','어느것','저기','저쪽','저것','그때','그럼','그러면','요만한걸','그래','그때','저것만큼','그저','이르기까지','할 줄 안다','할 힘이 있다','너','너희','당신','어찌','설마','차라리','할지언정','할지라도','할망정','할지언정','구토하다','게우다','토하다','메쓰겁다','옆사람','퉤','쳇','의거하여','근거하여','의해','따라','힘입어','그','다음','버금','두번째로','기타','첫번째로','나머지는','그중에서','견지에서','형식으로 쓰여','입장에서','위해서','단지','의해되다','하도록시키다','뿐만아니라','반대로','전후','전자','앞의것','잠시','잠깐','하면서','그렇지만','다음에','그러한즉','그런즉','남들','아무거나','어찌하든지','같다','비슷하다','예컨대','이럴정도로','어떻게','만약','만일','위에서 서술한바와같이','인 듯하다','하지 않는다면','만약에','무엇','무슨','어느','어떤','아래윗','조차','한데','그럼에도 불구하고','여전히','심지어','까지도','조차도','하지 않도록','않기 위하여','때','시각','무렵','시간','동안','어때','어떠한','하여금','네','예','우선','누구','누가 알겠는가','아무도','줄은모른다','줄은 몰랏다','하는 김에','겸사겸사','하는바','그런 까닭에','한 이유는','그러니','그러니까','때문에','그','너희','그들','너희들','타인','것','것들','너','위하여','공동으로','동시에','하기 위하여','어찌하여','무엇때문에','붕붕','윙','나','우리','엉엉','휘익','윙윙','오호','아하','어쨋든','만 못하다', '하기보다는','차라리','하는 편이 낫다','흐흐','놀라다','상대적으로 말하자면','마치','아니라면','쉿','그렇지 않으면','그렇지 않다면','안 그러면','아니었다면','하든지','아니면','이라면','좋아','알았어','하는것도','그만이다','어쩔수 없다','하나','일','일반적으로','일단','한켠으로는','오자마자','이렇게되면','이와같다면','전부','한마디','한항목','근거로','하기에','아울러','하지 않도록','않기 위해서','이르기까지','이 되다','로 인하여','까닭으로','이유만으로','이로 인하여','그래서','이 때문에','그러므로','그런 까닭에','알 수 있다','결론을 낼 수 있다','으로 인하여','있다','어떤것','관계가 있다','관련이 있다','연관되다','어떤것들','에 대해','이리하여','그리하여','여부','하기보다는','하느니','하면 할수록','운운','이러이러하다','하구나','하도다','다시말하면','다음으로','에 있다','에 달려 있다','우리','우리들','오히려','하기는한데','어떻게','어떻해','어찌됐어','어때','어째서','본대로','자','이','이쪽','여기','이것','이번','이렇게말하자면','이런','이러한','이와 같은','요만큼','요만한 것','얼마 안 되는 것','이만큼','이 정도의','이렇게 많은 것','이와 같다','이때','이렇구나','것과 같이','끼익','삐걱','따위','와 같은 사람들','부류의 사람들','왜냐하면','중의하나','오직','오로지','에 한하다','하기만 하면','도착하다','까지 미치다','도달하다','정도에 이르다','할 지경이다','결과에 이르다','관해서는','여러분','하고 있다','한 후','혼자','자기','자기집','자신','우에 종합한것과같이','총적으로 보면','총적으로 말하면','총적으로','대로 하다','으로서','참','그만이다','할 따름이다','쿵','탕탕','쾅쾅','둥둥','봐','봐라','아이야','아니','와아','응','아이','참나','년','월','일','령','영','일','이','삼','사','오','육','륙','칠','팔','구','이천육','이천칠','이천팔','이천구','하나','둘','셋','넷','다섯','여섯','일곱','여덟','아홉','령','영']

# 토큰 인덱싱 객체
v = CountVectorizer()

# 나이브 베이스 모델 객체
m1 = MultinomialNB()


# stopword 제거 함수
def get_couple(_words):
    global stopwords
    _words = [x for x in _words if x[0] not in stopwords]
    length = len(_words)
    for i in range(length - 1):
        yield _words[i][0], _words[i + 1][0]


# train 데이터가 모두 학습 될때까지 반복
while len(train_list) > 0:

    if len(train_list) > 37403:
        selected_list = random.sample(train_list, 37403)   # 메모리 사용률을 고려해서 조금 더 올려도 괜찮을 것 같음
        train_list = [index for index in train_list if index not in selected_list]
    else:
        selected_list = random.sample(train_list, len(train_list))
        train_list = [index for index in train_list if index not in selected_list]

    sampling_train_data = train_data.iloc[selected_list, :].reset_index(drop=True)

    tokenizer = Mecab()

    print("train data 토큰화")
    train_doc = [(tokenizer.pos(x), y) for x, y in tqdm(zip(sampling_train_data['text'], sampling_train_data['smishing']))]

    X_train = []  # text 값 리스트
    Y_train = []  # 각 text에 대한 smishing 값 리스트

    print("train data 토큰 필요없는 단어리스트 제거, 모형에 사용하기 위한 데이터 전처리")
    for lwords in train_doc:
        Y_train.append(lwords[1])

        temp = []
        for x, y in get_couple(lwords[0]):
            temp.append("{}.{}".format(x, y))

        X_train.append(" ".join(temp))

    # print("X_train Data : ")
    # print(X_train)

    v.fit(X_train)

    vec_x_train = v.transform(X_train).toarray()

    m1.fit(vec_x_train, Y_train)

# test 데이터 분석 시작
test_data['smishing'] = 2  # smishing 컬럼 추가

print("test data 토큰화")
test_doc = [(tokenizer.pos(x), y) for x, y in tqdm(zip(test_data['text'], test_data['smishing']))]

print("test data 토큰 필요없는 단어리스트 제거, 모형에 사용하기 위한 데이터 전처리")
X_test = []  # test데이터 text 값 리스트
for lwords in test_doc:

    temp = []
    for x, y in get_couple(lwords[0]):
        temp.append("{}.{}".format(x, y))

    X_test.append(" ".join(temp))


vec_x_test = v.transform(X_test).toarray()

y_test_pred1 = m1.predict_proba(vec_x_test)
y_test_pred1_one = [i[1] for i in y_test_pred1]

submission_data['smishing'] = y_test_pred1_one

submission_data.to_csv("sample_submission.csv", index=False)

