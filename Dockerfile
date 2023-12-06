# #FROM debian:latest as linfold_builder
# FROM ubuntu:18.04 as linfold_builder

# RUN apt-get update && apt-get install -y git gcc g++ make

# WORKDIR /app-docker
# RUN git clone https://github.com/LinearFold/LinearFold.git
# WORKDIR /app-docker/LinearFold
# RUN make && apt-get clean

#FROM debian:latest
FROM python:3.9-slim-buster

WORKDIR /app

COPY requirements.txt .

RUN apt-get update \
    && apt-get install -y python3 python3-pip nginx uwsgi uwsgi-plugin-python3 vim less \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip \
    && pip3 install -r requirements.txt

COPY nginx.conf /etc/nginx/nginx.conf

RUN ln -s /usr/bin/python3 /usr/bin/python && \
    ln -s /usr/bin/pip3 /usr/bin/pip

EXPOSE 80

ENTRYPOINT [""]
#ENTRYPOINT nginx -g "daemon on;" && uwsgi --ini uwsgi.ini

CMD [""]
#CMD [ "/bin/bash"]
