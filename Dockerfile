#FROM debian:latest as linfold_builder
FROM ubuntu:18.04 as linfold_builder
MAINTAINER Jingyi Wei "weijingy@stanford.edu"

RUN apt-get update && apt-get install -y git gcc g++ make

WORKDIR /app-docker
RUN git clone https://github.com/LinearFold/LinearFold.git
WORKDIR /app-docker/LinearFold
RUN make && apt-get clean

#FROM debian:latest
FROM ubuntu:18.04
COPY --from=linfold_builder /app-docker/LinearFold /app-docker/LinearFold
COPY . /app-docker

WORKDIR /app-docker
RUN apt-get update && \
    apt-get install -y python2.7 python3 python3-pip nginx-core nginx uwsgi uwsgi-plugin-python3 nano && \
    pip3 install --upgrade pip && \
    pip3 install -r requirements.txt && \
    apt-get clean

COPY nginx.conf /etc/nginx/nginx.conf

RUN ln -s /usr/bin/python2.7 /usr/bin/python2 && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    ln -s /usr/bin/pip3 /usr/bin/pip


EXPOSE 80

ENTRYPOINT [""]
#ENTRYPOINT nginx -g "daemon on;" && uwsgi --ini uwsgi.ini

CMD [""]
#CMD [ "/bin/bash"]
