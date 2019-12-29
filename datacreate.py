# 샘플 데이터 추출 코드

import os
import sys
import pandas as pd
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def sampling_data(seed):
    # under sampling 데이터 추출
    train_data = pd.read_csv(BASE_DIR+"/data/train.csv")

    random.seed(seed)
    train_nsm_list = list(train_data[train_data['smishing'] != 1].index)
    train_nsmishing = random.sample(train_nsm_list, 18703)
    train_smishing = list(train_data[train_data['smishing'] == 1].index)

    sampling_train_data = train_data.iloc[train_smishing + train_nsmishing, :].reset_index(drop=True)

    sampling_train_data.to_csv(BASE_DIR+"/data/test/under_train_data_"+str(seed)+".csv", index=False)

    # under sampling 데이터에서 80% train 데이터 20% test 데이터 추출
    under_train_data = pd.read_csv(BASE_DIR+"/data/test/under_train_data_"+str(seed)+".csv")

    index_list = list(under_train_data['smishing'].index)
    test_list = random.sample(list(under_train_data['smishing'].index), 7481)
    train_list = [index for index in index_list if index not in test_list]


    sample_train_data = under_train_data.iloc[train_list, :].reset_index(drop=True)
    sample_test_data = under_train_data.iloc[test_list, :].reset_index(drop=True)

    sample_train_data.to_csv(BASE_DIR+"/data/test/sample_train_data_"+str(seed)+".csv", index=False)
    sample_test_data.to_csv(BASE_DIR+"/data/test/sample_test_data_"+str(seed)+".csv", index=False)



