# Specified OS or runtime
FROM debian:latest

MAINTAINER Contextual Dynamics Lab <contextualdynamics@gmail.com>


RUN apt-get update && \
    apt-get install libgl1-mesa-glx -y
RUN apt-get install -y eatmydata


# install basic packages
RUN eatmydata apt-get install -y wget bzip2 ca-certificates \
    git \
    swig \
    mpich \
    pkg-config \
    libfontconfig1-dev \
    gcc \
    wget \
    curl \
    vim \
    nano


# install Anaconda python distribution
RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    wget --quiet https://repo.continuum.io/archive/Anaconda3-4.4.0-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh

# Setup anaconda path
ENV PATH /opt/conda/bin:$PATH


# Update pip and install various important python packages
RUN pip install --upgrade pip

RUN pip install --upgrade \
nose \
sphinx \
numpy \
scipy \
pandas \
matplotlib \
numba \
pykalman

# Due to behavior of pip>=10.1, we must use the `--ignore-installed` flag to
# install packages that were originally distutils installs (or rely on packages
# that were)
RUN pip install --upgrade --ignore-installed \
seaborn \
git+git://github.com/FIU-Neuro/brainconn.git \
hypertools \

RUN conda install scikit-learn

# add some useful directories as mirrors of directors in the same location on your computer
ADD data /data
ADD notebooks /notebooks


# Finally, expose a port from within the docker so we can use it to run
# jupyter notebooks
EXPOSE 9999