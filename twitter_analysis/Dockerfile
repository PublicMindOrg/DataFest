FROM python:3.9.12

RUN pip install tweepy

RUN pip install apscheduler

RUN pip install flask

WORKDIR /app

COPY . .

EXPOSE 3000

CMD [ "python", "-m" , "flask", "run", "--host=0.0.0.0"]
