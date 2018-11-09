#!/bin/bash
apt-get install -y --no-install-recommends \
      build-essential \
      cmake \
      git \
      libgoogle-glog-dev \
      libgtest-dev \
      libiomp-dev \
      libleveldb-dev \
      liblmdb-dev \
      libopencv-dev \
      libopenmpi-dev \
      libsnappy-dev \
      libprotobuf-dev \
      openmpi-bin \
      openmpi-doc \
      protobuf-compiler \
      python-dev \
      python-pip                          
pip install --user \
      future \
      numpy \
      protobuf


wget https://repo.anaconda.com/archive/Anaconda2-5.2.0-Linux-x86_64.sh
bash Anaconda2-5.2.0-Linux-x86_64.sh
export PATH="$HOME/anaconda2/bin:$PATH"

# install CUDA Toolkit v9.0
# instructions from https://developer.nvidia.com/cuda-downloads (linux -> x86_64 -> Ubuntu -> 16.04 -> deb)
CUDA_REPO_PKG="cuda-repo-ubuntu1604-9-0-local_9.0.176-1_amd64-deb"
wget https://developer.nvidia.com/compute/cuda/9.0/Prod/local_installers/${CUDA_REPO_PKG}
dpkg -i ${CUDA_REPO_PKG}
wget -qO - https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub | apt-key add -
# apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
apt-get update
apt-get -y install cuda-9-0

dpkg -i nccl-repo-ubuntu1604-2.2.13-ga-cuda9.0_1-1_amd64.deb
apt-get update
apt-get install libnccl2 libnccl-dev


# # install cuDNN v7.0
CUDNN_PKG="libcudnn7_7.2.1.38-1+cuda9.0_amd64.deb"
wget https://github.com/ashokpant/cudnn_archive/raw/master/v7.0/${CUDNN_PKG}
dpkg -i ${CUDNN_PKG}
apt-get update

# # install NVIDIA CUDA Profile Tools Interface ( libcupti-dev v9.0)
apt-get install cuda-command-line-tools-9-0

# # set environment variables
export PATH=${PATH}:/usr/local/cuda-9.0/bin
export CUDA_HOME=${CUDA_HOME}:/usr/local/cuda:/usr/local/cuda-9.0
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-9.0/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64




export PATH="$HOME/anaconda2/bin:$PATH"
conda install -c caffe2 pytorch-caffe2-cuda9.0-cudnn7 


