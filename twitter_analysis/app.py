import tweepy
from apscheduler.schedulers.background import BackgroundScheduler
from flask import Flask,redirect,url_for,jsonify, request
import csv  
import os 
app = Flask(__name__)
sched = BackgroundScheduler(daemon=True)

@app.route('/get_tweets/<string:query>')
def get_tweets(query):
    sched.add_job(get_all_tweets,'interval',args=[query],minutes=5,replace_existing=True)
    sched.start()
    return 'Hey'

def get_all_tweets(query):
    
    client = tweepy.Client(bearer_token='AAAAAAAAAAAAAAAAAAAAAGT8bgEAAAAAJZslobYs39BqCSVn%2BipiFNxH1ao%3D5osYxvUKqEteDrWV2eOFIQvABR1QCTy2RhHS450UwS6yTQuVlb')
    header = ['Tweet', 'Date', 'Language']
    filename = query+'.csv'
    with open(filename, 'a', encoding='UTF8') as f:
        writer = csv.writer(f)
        count=0
        for tweet in tweepy.Paginator(client.search_recent_tweets,query = query,
                tweet_fields= 'id,text,lang',
                                     max_results=100).flatten(limit=1000):
            # write the data
            print(tweet)
            count+=1
            print("Tweet added: ",count)
            data = [tweet.text,tweet.created_at,tweet.lang]
            writer.writerow(data)
    return "Total tweets added:"+str(count)

if __name__ == '__main__':
    # port = int(os.environ.get('PORT', 5000))
    app.run(debug=True)
    