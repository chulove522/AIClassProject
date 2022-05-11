FROM python:latest

WORKDIR /app

COPY . /app

RUN apt-get update

RUN pip install -r requirements.txt

EXPOSE 80

CMD [ "python", "app.py"]