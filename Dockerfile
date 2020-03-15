FROM 10.1-cudnn7-devel-ubuntu16.04

SHELL ["bash", "-c"] 

# install vim and git
RUN apt-get update && \
apt-get install vim \
git \
-y

# install python3.6-dev necessary for pycocotools
RUN apt-get install software-properties-common && \
add-apt-repository ppa:deadsnakes/ppa && \
apt-get update && \
apt-get install python3.6-dev \
virtualenv

CMD [ "tail", "-f", '/dev/null' ]




