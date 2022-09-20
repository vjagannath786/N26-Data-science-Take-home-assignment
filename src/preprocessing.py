import pandas as pd
from sklearn.preprocessing import LabelEncoder

import os
import config
import pickle



def preprocess():

    train = pd.read_csv(os.path.join(config.path,'2016-09-19_79351_training.csv'))
    mcc_cats = pd.read_csv(os.path.join(config.path,'mcc_group_definition.csv'))
    txn_types = pd.read_csv(os.path.join(config.path, 'transaction_types.csv'))

    txn_types.rename(columns={'explanation':'txn_explanations'}, inplace=True)

    train = train.merge(mcc_cats, on='mcc_group',how='left')

    train = train.merge(txn_types, left_on='transaction_type', right_on='type', how='left')


    train.drop(['type'],axis=1, inplace=True)




    le = LabelEncoder()

    combined = train

    #print(combined.shape)

    le.fit_transform(combined['user_id'])

    
    with open(os.path.join(config.model_path,'user_id.pkl'),'wb') as output:
        le_dict = dict(zip(le.classes_, le.transform(le.classes_)))
        pickle.dump(le_dict, output)

    
    le = LabelEncoder()

    le.fit_transform(combined['explanation'])

    with open(os.path.join(config.model_path,'categories.pkl'),'wb') as output:
        le_dict = dict(zip(le.classes_, le.transform(le.classes_)))
        pickle.dump(le_dict, output)


    




