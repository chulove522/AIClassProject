FROM python:3.8.13-alpine

RUN apk upgrade --no-cache

RUN mkdir -p /app
WORKDIR /app

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY . /app

EXPOSE 80

CMD [ "python", "app.py"]