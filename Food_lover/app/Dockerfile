FROM ubuntu:20.04

#RUN sed -i "s/archive.ubuntu./mirrors.aliyun./g" /etc/apt/sources.list
#RUN sed -i "s/deb.debian.org/mirrors.aliyun.com/g" /etc/apt/sources.list
#RUN sed -i "s/security.debian.org/mirrors.aliyun.com\/debian-security/g" /etc/apt/sources.list
COPY requirements.txt /app/requirements.txt
WORKDIR /app
RUN apt-get update -y && apt-get upgrade -y &&  apt-get install -y python3 python3-pip python3-dev
RUN DEBIAN_FRONTEND=noninteractive apt-get install pkg-config -y
RUN DEBIAN_FRONTEND=noninteractive apt-get install libmysqlclient-dev -y
RUN DEBIAN_FRONTEND=noninteractive apt-get install default-libmysqlclient-dev -y
RUN pip install -r requirements.txt

EXPOSE 5000
COPY .. /app


ENV FLASK_APP=app


ENV FLASK_RUN_PORT=5000
CMD [ "flask", "run", "--host", "0.0.0.0" ]
