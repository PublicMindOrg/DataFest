import pandas as pd
import os
import json
import time
import nltk
from nltk.tokenize import TweetTokenizer
import csv
import os
import string
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import preprocessor as p
import sys

def remove_punctuation(word):
    new_word = re.sub(r'[^\w\s]', '', (word))
    return new_word

english_vocab = set(w.lower() for w in nltk.corpus.words.words())
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
word_distributions={}
wkb_file = 'wkb.csv'
wkb_val_scores = {}

neg_corr_list = ['like','bed','respect','give','loyal','lol','aunt','love','daddy','goodnight','shower','truth','believe','understand','always','luv','bless','baby','sleep','mommy','me','song','faithful','wish','trust','avoid','against','taxes','delayed','ouch','shooting','captured','bill','dangerous','deadline','homework','tax','mortgage','accidentally','penalty','dammit','lynch','hunger','injuries','creepy','zero','fast','tough','nuclear','costs']

wkb_df = pd.read_csv(wkb_file)
for index, row in wkb_df.iterrows():
    word = row.iloc[1]
    word = word.lower()
    valence = row.iloc[3]
    word=lemmatizer.lemmatize(word)
    wkb_val_scores[word]=float(valence)

filespath='/data/Coronavirus-Tweets/with-state/'
savepath='/data/Coronavirus-Tweets/coronavirus-tweets-scored-latest/'
list_of_folders=os.listdir(filespath)
if '.DS_Store' in list_of_folders:
    list_of_folders.remove('.DS_Store')
for folder in list_of_folders:
    list_of_files = os.listdir(os.path.join(filespath,folder))
    if '.DS_Store' in list_of_files:
        list_of_files.remove('.DS_Store')
    for file in list_of_files:
        if 'coronavirus' not in file:
            list_of_files.remove(file)
    for filename in list_of_files:
        print(filename)
        valences=[]
        rel_words=[]
        df=pd.read_csv(os.path.join(filespath,folder,filename))
        df = df[df['text'].notna()]
        for i in range(len(df)):
            text=df['text'].iloc[i]
            number_of_words=0
            valence_sum = 0.0
            relevant_words=[]
            tweet_text = p.clean(text)
            tweet_text=tweet_text.strip('/n')
            tweet_text = tweet_text.lower()
            tweet_text = tweet_text.replace('\d+', '')
            tweet_text = tweet_text.replace('[^\w\s]', ' ').replace('\s\s+', ' ')
            if 'rt' in tweet_text:
                temp_text = tweet_text.split(' ')
                temp_text = temp_text[2:]
                tweet_text = ' '.join(temp_text)
            #translator = tweet_text.maketrans(string.punctuation.replace("\'",""), ' '*len(string.punctuation.replace("\'","")))
            #tweet_text = tweet_text.translate(translator)
            df['text'].iloc[i]=tweet_text
            w_tokenizer = TweetTokenizer()
            word_tokens = w_tokenizer.tokenize(tweet_text)

            filtered_sentence = [w.lower() for w in word_tokens if w.lower() not in neg_corr_list]
            lemmatized_words = []
            for w in filtered_sentence:
                word = lemmatizer.lemmatize(w.lower())
                if word !='' and word in wkb_val_scores and len(word)>3:
                    number_of_words +=1
                    valence_sum += float(wkb_val_scores[word])
                    relevant_words.append(word)
                    if word not in word_distributions:
                        word_distributions[word]=1
                    elif word in word_distributions:
                        word_distributions[word]+=1
            average_valence = 0.0
            if number_of_words > 0:
                average_valence = valence_sum/number_of_words
            valences.append(average_valence)
            rel_words.append(relevant_words)
        df['valence'] = valences
        df['rel_words']=rel_words
        if not os.path.exists(os.path.join(savepath,folder)):
            os.makedirs(os.path.join(savepath,folder))

        df.to_csv(os.path.join(savepath,folder,filename),index=False)
ddf = pd.DataFrame(word_distributions.items(), columns=['Word', 'Count'])
ddf.to_csv('word_distributions.csv')
