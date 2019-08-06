import pandas as pd 
import numpy as np 

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score

from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE, RandomOverSampler

train,test = pd.read_csv('cleantrain.csv'),pd.read_csv('testCopy.csv')

# tvec = TfidfVectorizer(stop_words=None, max_features=100000, ngram_range=(1, 3))
lr = LogisticRegression()
tvec = TfidfVectorizer(stop_words=None,max_features=100000, ngram_range=(1, 3))

#Calculate using Over Sampling
ROS_pipeline = make_pipeline(tvec, RandomOverSampler(random_state=777),lr)
SMOTE_pipeline = make_pipeline(tvec, SMOTE(random_state=777),lr)

def lr_cv(splits, train, test_data, pipeline, average_method):    
    #kfold = StratifiedKFold(n_splits=splits, shuffle=True, random_state=777)
    # train_text = train['text'].sample(n=2924, random_state=777)
    train_sentiment = train['sentiment'].sample(n=2924, random_state=777)
    accuracy = []
    precision = []
    recall = []
    f1 = []
    
    hash_ = list(test_data.unique_hash)
    lr_fit = pipeline.fit(train.text, train.sentiment)
    prediction = lr_fit.predict(test_data.text)
    scores = lr_fit.score(test_data.text, prediction)
    df = pd.DataFrame(list(zip(hash_,prediction)),columns=['unique_hash','sentiment'])

    # return df


    accuracy.append(scores * 100)
    precision.append(precision_score(train_sentiment, prediction, average=average_method)*100)
    #print('              negative    neutral     positive')
    #print('precision:',precision_score(train_sentiment, prediction, average=None))
    recall.append(recall_score(train_sentiment, prediction, average=average_method)*100)
    #print('recall:   ',recall_score(train_sentiment, prediction, average=None))
    f1.append(f1_score(train_sentiment, prediction, average=average_method)*100)
    print('f1 score: ',f1_score(train_sentiment, prediction, average=None))
    print('-'*50)

    print("accuracy: %.2f%% (+/- %.2f%%)" % (np.mean(accuracy), np.std(accuracy)))
    print("precision: %.2f%% (+/- %.2f%%)" % (np.mean(precision), np.std(precision)))
    print("recall: %.2f%% (+/- %.2f%%)" % (np.mean(recall), np.std(recall)))
    print("f1 score: %.2f%% (+/- %.2f%%)" % (np.mean(f1), np.std(f1)))
    # print(precision,end=' ')
    # print(recall, end=' ')
    # print(f1, end=' ')
    return df
#suppress warning
import warnings
warnings.filterwarnings('ignore')
#Random Over Sampler Pipeline
#lr_cv(5, train, test, ROS_pipeline, 'macro')

#Smote Pipeline
# lr_cv(3, train, test, SMOTE_pipeline, 'macro')

# #DownSampling
# from imblearn.under_sampling import NearMiss, RandomUnderSampler
# RUS_pipeline = make_pipeline(tvec, RandomUnderSampler(random_state=777),lr)
# NM1_pipeline = make_pipeline(tvec, NearMiss(ratio='not minority',random_state=777, version = 1),lr)
# NM2_pipeline = make_pipeline(tvec, NearMiss(ratio='not minority',random_state=777, version = 2),lr)

# #Rondam Under Sampling
# lr_cv(5, train, test, RUS_pipeline, 'macro')

# #Near Miss 1
# lr_cv(5, train, test, NM1_pipeline, 'macro')
# #Near Miss 2
# lr_cv(5, train, test, NM2_pipeline, 'macro')
