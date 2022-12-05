#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 22:56:45 2022

@author: dianapham
"""

import pandas as pd 
from vaderSentiment import SentimentIntensityAnalyzer

# GoEmotions Data
training = pd.read_table('data/train.tsv', header = None)
test = pd.read_table('data/test.tsv', header = None)
validation = pd.read_table('data/dev.tsv', header = None)
emotions = pd.concat([training, test, validation]).reset_index()
emotions.drop("index", inplace = True, axis = 1)
emotions.columns = ["text", "emotion_ids", "comment_id"]
 
# Creating multiple rows for rows with more than one emotion_ids
for row in list(range(0, len(emotions))):
    emotions['emotion_ids'][row] = emotions['emotion_ids'][row].split(',')
emotions = emotions.explode('emotion_ids').reset_index()
emotions.drop('index', inplace = True, axis = 1)

'''
# removing rows with emotion_ids that are less prominent (less than 1000 occurrences bc they seem nonexistent on the histogram)
emotions['emotion_ids'].value_counts()
for row in list(range(0, len(emotions))):
    if emotions['emotion_ids'][row] in ['8', '14', '24', '12', '19', '23', '21', '16']:
        emotions.drop([row], inplace = True, axis = 0)
emotions = emotions.reset_index()
emotions.drop('index', inplace = True, axis = 1) 
'''

'''
# REMOVING ROWS WITH MULTIPLE EMOTIONS FOR COMPARISON TO ORIGINAL ACCURACY
indexes = []
for row in list(range(0, len(emotions))):
    if ',' in emotions['emotion_ids'][row]:
        indexes.append(row)
emotions = emotions.drop(indexes, axis = 0)
emotions = emotions.reset_index()
emotions = emotions.drop(['index'], axis = 1)
#emotions = emotions[emotions['emotion_ids'].isin([','])]
'''

# Categorizing emotions_id as Positive, Negative, or Neutral
    # if there are multiple emotions for a row then categorize based on the first emotion 
emotion_list = pd.read_table("emotions.txt", header = None)
true_class = []
emotion = []
#        if '0' == emotions['emotion_ids'][i].split(',')[0]: #admiratioin (use this way if each row doesnt have only one emotion_ids)   
for i in list(range(0, len(emotions))):
    emotions['emotion_ids'][i] = int(emotions['emotion_ids'][i])
    if 0 == emotions['emotion_ids'][i]: #admiration
        true_class.append('Positive')
        emotion.append('admiration')
    elif 1 == emotions['emotion_ids'][i]: #amusement
        true_class.append('Positive')
        emotion.append('amusement')
    elif 2 == emotions['emotion_ids'][i]: #anger
        true_class.append('Negative')
        emotion.append('anger')
    elif 3 == emotions['emotion_ids'][i]: #annoyance
        true_class.append('Negative')
        emotion.append('annoyance')
    elif 4 == emotions['emotion_ids'][i]: #approval
        true_class.append('Positive')
        emotion.append('approval')
    elif 5 == emotions['emotion_ids'][i]: #caring
        true_class.append('Positive')
        emotion.append('caring')
    elif 6 == emotions['emotion_ids'][i]: #confusion
        true_class.append('Negative')
        emotion.append('confusion')
    elif 7 == emotions['emotion_ids'][i]: #curiosity
        true_class.append('Positive')
        emotion.append('curiosity')
    elif 8 == emotions['emotion_ids'][i]: #desire
        true_class.append('Positive')
        emotion.append('desire')
    elif 9 == emotions['emotion_ids'][i]: #disappointment
        true_class.append('Negative')
        emotion.append('disappointment')
    elif 10 == emotions['emotion_ids'][i]: #disapproval
        true_class.append('Negative')
        emotion.append('disapproval')
    elif 11 == emotions['emotion_ids'][i]: #disgust
        true_class.append('Negative')
        emotion.append('disgust')
    elif 12 == emotions['emotion_ids'][i]: #embarassment
        true_class.append('Negative')
        emotion.append('embarassment')
    elif 13 == emotions['emotion_ids'][i]: #excitement
        true_class.append('Positive')
        emotion.append('excitement')
    elif 14 == emotions['emotion_ids'][i]: #fear
        true_class.append('Negative')
        emotion.append('fear')
    elif 15 == emotions['emotion_ids'][i]: #gratitude
        true_class.append('Positive')
        emotion.append('gratitude')
    elif 16 == emotions['emotion_ids'][i]: #grief
        true_class.append('Negative')
        emotion.append('grief')
    elif 17 == emotions['emotion_ids'][i]: #joy
        true_class.append('Positive')
        emotion.append('joy')
    elif 18 == emotions['emotion_ids'][i]: #love
        true_class.append('Positive')
        emotion.append('love')
    elif 19 == emotions['emotion_ids'][i]: #nervousness
        true_class.append('Negative')
        emotion.append('nervousness')
    elif 20 == emotions['emotion_ids'][i]: #optimism
        true_class.append('Positive')
        emotion.append('optimism')
    elif 21 == emotions['emotion_ids'][i]: #pride
        true_class.append('Positive')
        emotion.append('pride')
    elif 22 == emotions['emotion_ids'][i]: #realization
        true_class.append('Neutral')
        emotion.append('realization')
    elif 23 == emotions['emotion_ids'][i]: #relief
        true_class.append('Positive')
        emotion.append('relief')
    elif 24 == emotions['emotion_ids'][i]: #remorse
        true_class.append('Negative')
        emotion.append('remorse')
    elif 25 == emotions['emotion_ids'][i]: #sadness
        true_class.append('Negative')
        emotion.append('sadness')
    elif 26 == emotions['emotion_ids'][i]: #surprise
        true_class.append('Neutral')
        emotion.append('surprise')
    elif 27 == emotions['emotion_ids'][i]: # neutral
        true_class.append('Neutral')
        emotion.append('neutral')
    else:
        true_class.append('Neutral')
        emotion.append('neutral')
emotions['emotion'] = emotion
emotions['true_class'] = true_class
#emotions['emotion_ids'].plot.hist(grid=True, bins = 28) # histogram of emotion_ids (placing here bc they become numeric here)
 
# running vader sentiment analysis    
sentences = list(emotions['text'])
analyzer = SentimentIntensityAnalyzer()

for sentence in sentences:
    vs = analyzer.polarity_scores(sentence)
    print("{:-<65} {}".format(sentence, str(vs)))
    
# adding compound vader score to the main dataset
cs = []
for row in range(len(emotions)):
    cs.append(analyzer.polarity_scores(emotions['text'].iloc[row])['compound'])
    
emotions['compound_vader_score'] = cs
#emotions = emotions[(emotions[['compound_vader_score']] != 0).all(axis=1)].reset_index(drop=True)

# adding a classification of Positive or Negative based on the compound vader score
classification = []
for i in list(range(0, len(emotions))):
    if emotions['compound_vader_score'][i] >= 0.05:
        classification.append("Positive")
    elif emotions['compound_vader_score'][i] <= -0.05:
        classification.append("Negative")
    else:
        classification.append("Neutral")

emotions['class'] = classification

# barchart of classifications
import matplotlib.pyplot as plt
emotions['class'].value_counts()
classes = {"Negative": emotions['class'].value_counts()[1], "Neutral": emotions['class'].value_counts()[2], "Positive": emotions['class'].value_counts()[0]}
plt.bar(list(classes.keys()), list(classes.values()), color = ['maroon', 'gold', 'steelblue'])
plt.xlabel("Sentiment Classification")
plt.ylabel("Number of cases")
plt.title("Sentiment Classification of GoEmotions Dataset")
plt.show()

# histogram of compound scores
emotions[['compound_vader_score']].plot.hist(color = 'gold', bins = 15)
plt.xlabel("Compound Score")
plt.ylabel("Frequency")
plt.title("Distribution of VADER Compound Scores")
plt.show()

'''
# calculating accuracy  <-- not relevant since classes are imbalanced
correct = 0
for i in list(range(0, len(emotions))):
    if emotions['class'][i] == emotions['true_class'][i]:
        correct += 1
print(correct/len(emotions)) 
    # NOTE: this number is dependent on how you categorize the emotions.
    # I looked up each emotion to see if it was a Positive or Negative emotion 
    # then categorized each row by the first emotion listed in emotion_ids
'''
    
# Creating  a confusion matrix,which compares the y_test and y_pred
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(emotions['true_class'], emotions['class'])
# Creating a dataframe for a array-formatted Confusion matrix,so it will be easy for plotting.
cm_df = pd.DataFrame(cm, index = ['Neutral','Negative','Positive'], columns = ['Neutral','Negative','Positive'])   

## PERFORMANCE METRICS
# Neutral 
print('\nNEUTRAL CLASS:')
# calculating true positive
neu_tp = cm_df['Neutral'][0]
print('True Positive:', neu_tp)
# calculating true negative
neu_tn = cm_df['Negative'][1] + cm_df['Negative'][2] + cm_df['Positive'][1] + cm_df['Positive'][2]
print('True Negative:', neu_tn)
# calculating false positive
neu_fp = cm_df['Neutral'][1] + cm_df['Neutral'][2]
print('False Positive:', neu_fp)
# calculating false negative
neu_fn = cm_df['Negative'][0] + cm_df['Positive'][0]
print('False Negative:',neu_fn)
#calculating precision
neu_precision = neu_tp / (neu_tp + neu_fp)
print('Precision:', neu_precision)
#calculating recall
neu_recall = neu_tp / (neu_tp + neu_fn)
print('Recall:', neu_recall)
# calculating f1 score
print('F1 Score:', (2 * neu_precision * neu_recall) / (neu_precision + neu_recall))
# calculating accuracy
print('Accuracy:', (neu_tp + neu_tn) / (neu_tp + neu_tn + neu_fp +  neu_fn))

# Negative
print('\nNEGATIVE CLASS:')
# calculating true positive
neg_tp = cm_df['Negative'][1]
print('True Positive:', neg_tp)
# calculating true negative
neg_tn = cm_df['Neutral'][0] + cm_df['Neutral'][2] + cm_df['Positive'][0] + cm_df['Positive'][2]
print('True Negative:', neg_tn)
# calculating false positive
neg_fp = cm_df['Negative'][0] + cm_df['Negative'][2]
print('False Positive:', neg_fp)
# calculating false negative
neg_fn = cm_df['Neutral'][1] + cm_df['Positive'][1]
print('False Negative:', neg_fn)
# calculating precision
neg_precision = neg_tp / (neg_tp + neg_fp)
print('Precision:', neg_precision)
#calculating recall
neg_recall = neg_tp / (neg_tp + neg_fn)
print('Recall:', neg_recall)
# calculating f1 score
print('F1 Score:', (2 * neg_precision * neg_recall) / (neg_precision + neg_recall))
# calculating accuracy
print('Accuracy:', (neg_tp + neg_tn) / (neg_tp + neg_tn + neg_fp + neg_fn))

# Positive
print('\nPOSITIVE CLASS:')
# calculating true positive
pos_tp = cm_df['Positive'][2]
print('True Positive:', pos_tp)
# calculating true negative
pos_tn = cm_df['Neutral'][0] + cm_df['Neutral'][1] + cm_df['Negative'][0] + cm_df['Negative'][1]
print('True Negative:', pos_tn)
# calculating false positive
pos_fp = cm_df['Positive'][0] + cm_df['Positive'][1]
print('False Positive:', pos_fp)
# calculating false negative
pos_fn = cm_df['Neutral'][2] + cm_df['Negative'][2]
print('False Negative:', pos_fn)
# calculating precision
pos_precision = pos_tp / (pos_tp + pos_fp)
print('Precision:', pos_precision)
#calculating recall
pos_recall = pos_tp / (pos_tp + pos_fn)
print('Recall:', pos_recall)
# calculating f1 score
print('F1 Score:', (2 * pos_precision * pos_recall) / (pos_precision + pos_recall))
# calculating accuracy
print('Accuracy:', (pos_tp + pos_tn) / (pos_tp + pos_tn + pos_fp + pos_fn))

# creating column of predictions that match the 'results_agg' file in the datafest google drive
emotions_predict = emotions[['comment_id', 'class', 'compound_vader_score']]
results_agg = pd.read_csv('results_agg - goemotions.csv')
df_join = pd.merge(results_agg, emotions_predict, how='inner', on = 'comment_id')
class_num = []
for i in list(range(0, len(df_join))):
    if df_join['class'][i] == 'Negative':
        class_num.append(1)
    elif df_join['class'][i] == 'Neutral':
        class_num.append(2)
    else:
        class_num.append(3)
df_join['class_num'] = class_num
#df_join.to_csv('results_agg_vader.csv') # writing the data frame to a csv file

## CORRELATION MATRIX FOR VADER CLASS & EMOTION (DIDN'T WORK)
# reference link: https://blog.knoldus.com/how-to-find-correlation-value-of-categorical-variables/
#from dython.nominal import associations
#from dython.nominal import identify_nominal_columns
#categorical_features = identify_nominal_columns(emotions)
#complete_correlation= associations(emotions, filename= 'complete_correlation.png', figsize=(10,10))   
#emotions_complete_corr=complete_correlation['corr']
#emotions_complete_corr.dropna(axis=1, how='all').dropna(axis=0, how='all').style.background_gradient(cmap='coolwarm', axis=None).set_precision(2)
 






