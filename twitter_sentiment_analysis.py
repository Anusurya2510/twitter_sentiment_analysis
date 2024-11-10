# -*- coding: utf-8 -*-
"""twitter_sentiment_analysis.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1bHC_vv5oB2QQMi5XEdNTe67hqkm4PvRC
"""

# Commented out IPython magic to ensure Python compatibility.
import re
import nltk
import string
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords

pd.set_option("display.max_colwidth", 200)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# %matplotlib inline

train = pd.read_csv('/content/train_E6oV3lV.csv')
test = pd.read_csv('/content/test_tweets_anuFYb8.csv')

# Text preprocessing - 1. Data Inspection , 2. Data Cleaning
train[train['label'] == 0].head(10)

train[train['label'] == 1].head(10)

print(train.shape)
print(test.shape)

train['label'].value_counts()

length_train = train['tweet'].str.len()
length_test = test['tweet'].str.len()
plt.hist(length_train, bins = 20, label = "train_tweets")
plt.hist(length_test, bins = 20, label = "test_tweets")
plt.legend()
plt.show()

#combining datasets for preprocessing data
combi = pd.concat([train, test], ignore_index=True)
combi.shape

#remove unwanted patterns
def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for word in r:
        input_txt = re.sub(word, "", input_txt)
    return input_txt

#removing twitter handles (@user)
combi['tidy_tweet'] = np.vectorize(remove_pattern)(combi['tweet'], "@[\w]*")
combi.head()

#removing punctuations , numbers, and special characters
combi["tidy_tweet"] = combi["tidy_tweet"].str.replace("[^a-zA-Z#]", " ")
combi.head()

#removing short words
combi["tidy_tweet"] = combi["tidy_tweet"].apply(lambda x: " ".join([w for w in x.split() if len(w)>3]))
combi.head()

#text normalization
tokenized_tweet = combi["tidy_tweet"].apply(lambda x: x.split())
print(tokenized_tweet.head())
from nltk.stem.porter import *
stemmer = PorterStemmer()

tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) # stemming
# print(tokenized_tweet.head())

for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = " ".join(tokenized_tweet[i])

combi["tidy_tweet"] = tokenized_tweet

#understanding the common words used in tweets: WordCloud
import wordcloud

all_words = " ".join([text for text in combi["tidy_tweet"]])

wordcloud = wordcloud.WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

#words in non racist/sexist tweets
import wordcloud
normal_words = " ".join([text for text in combi["tidy_tweet"][combi["label"] == 0]])
wordcloud = wordcloud.WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(normal_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# racist/sexist words
import wordcloud
negative_words = " ".join([text for text in combi["tidy_tweet"][combi["label"] == 1]])
wordcloud = wordcloud.WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(negative_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

#understanding the impact of hashtags on tweet sentiment

# extract hashtags
def hashtag_extract(x):
    hashtags = []
    for i in x:
        ht = re.findall(r"#(\w+)", i)
        hashtags.append(ht)
    return hashtags

# extracting hashtags from non-racist/sexist tweets
HT_regular = hashtag_extract(combi["tidy_tweet"][combi["label"] == 0])

# extracting hashtags from racist/sexist tweets
HT_negative = hashtag_extract(combi["tidy_tweet"][combi["label"] == 1])

# unnesting list
HT_regular = sum(HT_regular, [])
HT_negative = sum(HT_negative, [])

a = nltk.FreqDist(HT_regular)
d = pd.DataFrame({"Hashtag": list(a.keys()), "Count": list(a.values())})
#selecting top 20 most frequent hashtags
d = d.nlargest(columns = "Count", n = 20)
plt.figure(figsize = (16, 5))
ax = sns.barplot(data = d, x = "Hashtag", y = "Count")
ax.set(ylabel = "Count")
plt.show()

b = nltk.FreqDist(HT_negative)
e = pd.DataFrame({"Hashtag": list(b.keys()), "Count": list(b.values())})
e = e.nlargest(columns = "Count", n = 20)
plt.figure(figsize = (16, 5))
ax = sns.barplot(data = e, x = "Hashtag", y = "Count")
ax.set(ylabel = "Count")
plt.show()

#analysing preprocessed data , needs to be converted into features
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import gensim
from gensim.models import Word2Vec
#bag of words features
bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
bow = bow_vectorizer.fit_transform(combi['tidy_tweet'])
bow.shape

#TF-IDF FEATURES
# it works by penalising the common words by assigning them lower weights while giving importance to words which are rare
# in entire corpus but appear in good numbers in few documents
# TF-IDF = TF*IDF
# TF = (no of times t appears in a doc)/(no of terms in doc)
# IDF = log(N/n) where N is no of docs , n is no of docs t has appeared in

tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(combi['tidy_tweet'])
tfidf.shape

import gensim
from gensim.models import Word2Vec
# Word2Vec features
# word embeddings : representing words as vectors
# redefines high dimensional word features into low dimensional feature vectors by preserving contextual similarity in corpus
tokenized_tweet = combi['tidy_tweet'].apply(lambda x: x.split())
model_w2v = Word2Vec(
    sentences=tokenized_tweet,  # Changed 'tokenized_tweet' to 'sentences'
    vector_size=200,  #desired no of features / independent var
    window=5,  # context window size
    min_count=2,
    sg=1,  # skip gram model
    negative=10,  # for negative sampling
    workers=2,  # no of cores
    seed=34
)

model_w2v.train(tokenized_tweet, total_examples=len(combi['tidy_tweet']), epochs=20)
# model_w2v.wv.most_similar(positive="dinner")
# model_w2v.wv.most_similar(positive="trump")
model_w2v.wv['food'] # vector representation of word from our corpus
len(model_w2v.wv['food'])

from itertools import count
#preparing vectors for tweets
def word_vector(tokens, size):
  vec = np.zeros(size).reshape((1, size))
  count = 0
  for word in tokens:
    try:
      vec += model_w2v.wv[word].reshape((1, size))
      count += 1
    except KeyError:
      continue
  if count != 0:
    vec /= count
  return vec


wordvec_arrays = np.zeros((len(tokenized_tweet), 200))
for i in range (len(tokenized_tweet)):
  wordvec_arrays[i,:] = word_vector(tokenized_tweet[i], 200)

wordvec_df = pd.DataFrame(wordvec_arrays)
wordvec_df.shape

#doc2vec embedding
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
from gensim.models.doc2vec import TaggedDocument

def add_label(twt):
  output = []
  for i, s in zip(twt.index, twt):
    output.append(TaggedDocument(s, ["tweet_" + str(i)]))
  return output

labeled_tweets = add_label(combi['tidy_tweet'])
labeled_tweets[:6]

model_d2v = gensim.models.doc2vec.Doc2Vec( dm = 1 , #distributed memory
                                          dm_mean = 1 ,
                                          window = 5,
                                          negative = 7,#if >0 then negative sampling
                                          min_count = 5,
                                          workers = 3,
                                          alpha = 0.1,
                                          seed = 23
                                          )

model_d2v.build_vocab([i for i in tqdm(labeled_tweets)])
model_d2v.train(labeled_tweets, total_examples=len(combi['tidy_tweet']), epochs=15)

docvec_arrays = np.zeros((len(combi), 100))
for i in range(len(combi)):
   docvec_arrays[i, :] = model_d2v.dv[i]

docvec_df = pd.DataFrame(docvec_arrays)
docvec_df.shape

# algos to build models : 1. logistic regression , 2. svm , 3. RandomForest, 4. XGBoost
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
#fit logistic regression models on BOW features
train_bow = bow[:31962, :]
test_bow = bow[31962:, :]

from sklearn.linear_model import LogisticRegression
xtrain_bow, xvalid_bow, ytrain_bow, yvalid_bow = train_test_split(train_bow, train['label'], random_state=42, test_size=0.3)

lreg = LogisticRegression()
lreg.fit(xtrain_bow, ytrain_bow)

prediction = lreg.predict_proba(xvalid_bow)
prediction_int = prediction[:,1] >= 0.3
prediction_int = prediction_int.astype(int)


test_pred = lreg.predict_proba(test_bow)
test_pred_int = test_pred[:,1] >= 0.3
test_pred_int = test_pred_int.astype(int)
test['label'] = test_pred_int
submission = test[['id','label']]
submission.to_csv('sub_lreg_bow.csv', index=False)

f1_score(yvalid_bow, prediction_int)

train_tfidf = tfidf[:31962, :]
test_tfidf = tfidf[31962:, : ]
xtrain_tfidf = train_tfidf[ytrain_bow.index]
xvalid_tfidf = train_tfidf[yvalid_bow.index]

lreg.fit(xtrain_tfidf, ytrain_bow)
prediction = lreg.predict_proba(xvalid_tfidf)
prediction_int = prediction[:,1] >= 0.3
prediction_int = prediction_int.astype(int)
f1_score(yvalid_bow, prediction_int)

train_w2v = wordvec_df.iloc[:31962,:]
test_w2v = wordvec_df.iloc[31962:,:]

xtrain_w2v = train_w2v.iloc[ytrain_bow.index,:]
xvalid_w2v = train_w2v.iloc[yvalid_bow.index,:]

lreg.fit(xtrain_w2v, ytrain_bow)
prediction = lreg.predict_proba(xvalid_w2v)
prediction_int = prediction[:,1] >= 0.3
prediction_int = prediction_int.astype(int)
f1_score(yvalid_bow, prediction_int)

train_d2v = docvec_df.iloc[:31962,:]
test_d2v = docvec_df.iloc[31962:, :]

xtrain_d2v = train_d2v.iloc[ytrain_bow.index,:]
xvalid_d2v = train_d2v.iloc[yvalid_bow.index,:]

lreg.fit(xtrain_d2v, ytrain_bow)
prediction = lreg.predict_proba(xvalid_d2v)
prediction_int = prediction[:,1] >= 0.3
prediction_int = prediction_int.astype(int)
f1_score(yvalid_bow, prediction_int)

from sklearn import svm
svc = svm.SVC(kernel='linear', C=1, probability=True).fit(xtrain_bow, ytrain_bow)
prediction = svc.predict_proba(xvalid_bow)
prediction_int = prediction[:,1] >= 0.3
prediction_int = prediction_int.astype(int)
f1_score(yvalid_bow, prediction_int)

test_pred = svc.predict_proba(test_bow)
test_pred_int = test_pred[:,1] >= 0.3
test_pred_int = test_pred_int.astype(int)
test['label'] = test_pred_int
submission = test[['id','label']]
submission.to_csv('sub_svm_bow.csv', index=False)

svc = svm.SVC(kernel='linear', C=1, probability=True).fit(xtrain_tfidf, ytrain_bow)
prediction = svc.predict_proba(xvalid_tfidf)
prediction_int = prediction[:,1] >= 0
prediction_int = prediction_int.astype(int)
f1_score(yvalid_bow, prediction_int)

svc = svm.SVC(kernel='linear', C=1, probability=True).fit(xtrain_w2v, ytrain_bow)
prediction = svc.predict_proba(xvalid_w2v)
prediction_int = prediction[:,1] >= 0.3
prediction_int = prediction_int.astype(int)
f1_score(yvalid_bow, prediction_int)

svc = svm.SVC(kernel='linear', C=1, probability=True).fit(xtrain_d2v, ytrain_bow)
prediction = svc.predict_proba(xvalid_d2v)
prediction_int = prediction[:,1] >= 0.3
prediction_int = prediction_int.astype(int)
f1_score(yvalid_bow, prediction_int)

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=400, random_state=11).fit(xtrain_bow, ytrain_bow)
prediction = rf.predict(xvalid_bow)

test_pred = rf.predict_proba(test_bow)
test['label'] = test_pred
submission = test[['id','label']]
submission.to_csv('sub_rf_bow.csv', index=False)

f1_score(yvalid_bow, prediction)

rf = RandomForestClassifier(n_estimators=400, random_state=11).fit(xtrain_tfidf, ytrain_bow)
prediction = rf.predict(xvalid_tfidf)
f1_score(yvalid_bow, prediction)

rf = RandomForestClassifier(n_estimators=400, random_state=11).fit(xtrain_w2v, ytrain_bow)
prediction = rf.predict(xvalid_w2v)
f1_score(yvalid_bow, prediction)

rf = RandomForestClassifier(n_estimators=400, random_state=11).fit(xtrain_d2v, ytrain_bow)
prediction = rf.predict(xvalid_d2v)
f1_score(yvalid_bow, prediction)

from xgboost import XGBClassifier
xgb = XGBClassifier(max_depth= 6,n_estimators=1000).fit(xtrain_bow, ytrain_bow)
prediction = xgb.predict(xvalid_bow)
f1_score(yvalid_bow, prediction)

xgb_model = XGBClassifier(max_depth= 6,n_estimators=1000).fit(xtrain_tfidf, ytrain_bow)
prediction = xgb_model.predict(xvalid_tfidf)
test_pred = xgb_model.predict(test_tfidf)
test['label'] = test_pred
submission = test[['id','label']]
submission.to_csv('sub_xgb_tfidf.csv', index=False)
f1_score(yvalid_bow, prediction)

xgb = XGBClassifier(max_depth= 6,n_estimators=1000).fit(xtrain_tfidf, ytrain_bow)
prediction = xgb.predict(xvalid_tfidf)
f1_score(yvalid_bow, prediction)

xgb = XGBClassifier(max_depth= 6,n_estimators=1000).fit(xtrain_w2v, ytrain_bow)
prediction = xgb.predict(xvalid_w2v)
f1_score(yvalid_bow, prediction)

xgb = XGBClassifier(max_depth= 6,n_estimators=1000).fit(xtrain_d2v, ytrain_bow)
prediction = xgb.predict(xvalid_d2v)
f1_score(yvalid_bow, prediction)

import xgboost as xgb

dtrain = xgb.DMatrix(xtrain_w2v, label=ytrain_bow)
dvalid = xgb.DMatrix(xvalid_w2v, label=yvalid_bow)
dtest = xgb.DMatrix(test_w2v)

params = {
    "objective": "binary:logistic",
    "max_depth": 6,
    "min_child_weight": 1,
    "eta": 0.3,
    "subsample": 1,
    "colsample_bytree": 1

}

def custom_eval(preds, dtrain):
    labels = dtrain.get_label()
    preds = (preds >= 0.3).astype(int)
    return [("f1_score", f1_score(labels, preds))]

#parameter tuning
gridsearch_params = [
    (max_depth, min_child_weight)
    for max_depth in range(9,12)
    for min_child_weight in range(5,8)
]

max_f1 = 0
best_params = None
for max_depth, min_child_weight in gridsearch_params:
    print("CV with max_depth={}, min_child_weight = {}".format(
        max_depth,
        min_child_weight ))

    params['max_depth'] = max_depth
    params['min_child_weight'] = min_child_weight

    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=1000,
        seed=16,
        nfold = 5,
        early_stopping_rounds=10,
        feval = custom_eval
    )

#finding best f1 score
mean_f1 = cv_results['test-f1_score-mean'].max()
boost_rounds = cv_results['test-f1_score-mean'].argmax()

print("max f1 score {} for {} rounds".format(mean_f1, boost_rounds))
if mean_f1 > max_f1:
    max_f1 = mean_f1
    best_params = (max_depth, min_child_weight)
if best_params is not None:
    print("Best params: {}, {}, F1: {}".format(best_params[0], best_params[1], max_f1))
else:
    print("No better parameters found during grid search.")

params["max_depth"] = 8
params["min_child_weight"] = 6

gridsearch_params = [(subsample , colsample)
                     for subsample in [i/10. for i in range(5,10)]
                     for colsample in [i/10. for i in range(5,10)]
                     ]

max_f1 = 0
best_params = None
for subsample, colsample in gridsearch_params:
    print("CV with subsample={}, colsample={}".format(
        subsample,
        colsample
    ))

params["subsample"] = subsample
params["colsample"] = colsample
cv_results = xgb.cv(
    params,
    dtrain,
    feval = custom_eval,
    num_boost_round = 200,
    early_stopping_rounds = 10,
    seed = 16,
    nfold = 5,
    maximize = True
)

mean_f1 = cv_results["test-f1_score-mean"].max()
boost_rounds = cv_results["test-f1_score-mean"].argmax()
print("\tF1 {} for {} rounds".format(mean_f1, boost_rounds))
if mean_f1 > max_f1:
    max_f1 = mean_f1
    best_params = (subsample, colsample)

print("Best params: {}, {}, F1: {}".format(
    best_params[0],
    best_params[1],
    max_f1
))

params["subsample"] = 0.9
params["colsample_bytree"] = 0.7

max_f1 = 0
best_params = None
for eta in [.3, .2, .1, .05, .01, .005]:
    print("CV with eta={}".format(eta))
    params["eta"] = eta

cv_results = xgb.cv(
    params,
    dtrain,
    feval = custom_eval,
    num_boost_round = 1000,
    early_stopping_rounds = 20,
    seed = 16,
    nfold = 5,
    maximize = True
)

mean_f1 = cv_results["test-f1_score-mean"].max()
boost_rounds = cv_results["test-f1_score-mean"].argmax()
print("\tF1 {} for {} rounds".format(mean_f1, boost_rounds))
if mean_f1 > max_f1:
    max_f1 = mean_f1
    best_params = eta

print("Best params: {}, F1: {}".format(best_params, max_f1))

params= {'colsample': 0.9,
    'colsample_bytree': 0.5,
    'eta': 0.1,
    'max_depth': 8,
    'min_child_weight': 6,
    'objective': 'binary:logistic',
    'subsample': 0.9}

xgb_model = xgb.train(params , dtrain, feval =custom_eval, num_boost_round = 1000, maximize=True,
                      evals=[(dvalid, 'Validation')], early_stopping_rounds=10)

test_pred = xgb_model.predict(dtest)
test['label'] = test_pred
submission = test[['id','label']]
submission.to_csv('sub_xgb_w2v_finetuned.csv', index=False)