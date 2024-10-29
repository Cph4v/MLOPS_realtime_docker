FROM python:3.10-slim

RUN apt-get update && apt-get install -y git
RUN apt-get install -y zip
RUN apt-get install -y tmux
RUN apt-get install -y sudo
RUN apt-get install -y nano
RUN apt-get install -y cron

RUN mkdir /root/app
WORKDIR /root/app

# handel git ssh:
RUN mkdir -p ~/.ssh && ln -s /run/secrets/user_ssh_key ~/.ssh/id_rsa

# install dependencies
COPY ./requirements.txt .

RUN pip install mt5linux
RUN pip install -r requirements.txt
RUN pip install --upgrade pip 
RUN pip install python-telegram-bot==13.15



