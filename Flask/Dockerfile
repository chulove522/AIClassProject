FROM python:latest

WORKDIR /app

COPY . /app

RUN apt-get update && apt-get -y upgrade

RUN pip install -r requirements.txt

RUN pip3 install pymongo[srv]

EXPOSE 80

CMD [ "python", "main.py"]