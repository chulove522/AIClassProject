FROM python:3.8.13-buster

WORKDIR /app

COPY . /app

RUN apt-get update && apt-get -y upgrade

RUN pip install -r requirements.txt

EXPOSE 80

CMD [ "python", "main.py"]