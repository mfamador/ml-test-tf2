FROM python:3.8-slim

RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

RUN apt-get update && apt-get install -y git gcc
COPY requirements.txt /usr/src/app/
RUN pip3 install --no-cache-dir -r requirements.txt
RUN apt-get purge -y gcc git

COPY . /usr/src/app

ENV PYTHONPATH /usr/lib/python:/usr/src/app/src:/usr/src/app/src/retailerpredictor

# Gunicorn listening port
EXPOSE 8080

CMD [ "gunicorn", "-c", "gunicorn.cfg", "app:app"]
