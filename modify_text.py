# 맞춤법 검사 코드

import os
from hanspell import spell_checker
import pandas as pd
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

modified_text = pd.read_csv("modified_text.csv")
train_data = pd.read_csv(BASE_DIR+"/data/train.csv")    # 스미싱 데이터(1) : 18703개, 스미싱 데이터 X (0) : 277242개
id_list = []
text_list = []
count = 0

train_data['checked'] = 2
train_data['result'] = 3
result_list = []

for i in range(len(train_data)):
    try:
        result = spell_checker.check(train_data["text"][i])
        resultTy = result.as_dict()

        if resultTy['errors'] != 0 and resultTy['result'] is True:
            result_list.append(resultTy['result'])
            text_list.append(resultTy['checked'])
            count += 1
        else:
            result_list.append(resultTy['result'])
            text_list.append("--")
            count += 1

        print(count)
    except Exception as ex:
        count += 1
        print("check error : ", ex)
        print("error id : " + str(train_data['id'][i]))
        result_list.append(False)
        text_list.append("error")
        print(count)

train_data['checked'] = text_list
train_data['result'] = result_list

# modified_text['id'] = id_list
# modified_text['text'] = text_list
#
# modified_text.to_csv("modified_result.csv", index=False)

train_data.to_csv("modified_result.csv", index=False)



