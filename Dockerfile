# CUDA 및 CUDNN이 포함된 NVIDIA Docker 베이스 이미지
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu20.04

# 기본 환경 설정
ENV TZ=Asia/Seoul
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ARG DEBIAN_FRONTEND=noninteractive

# 필수 시스템 패키지 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    build-essential libgl1-mesa-glx libglib2.0-0 \
    git wget curl unzip \
    && rm -rf /var/lib/apt/lists/*

# PyTorch 설치
RUN pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

# Python 패키지 설치
COPY requirements.txt /workspace/requirements.txt
RUN pip install -r /workspace/requirements.txt

# dift 설치 (GitHub 저장소에서 설치)
# RUN pip install git+https://github.com/Tsingularity/dift.git

# taming-transformers 설치
RUN pip install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers

# CLIP 설치
RUN pip install -e git+https://github.com/openai/CLIP.git@main#egg=clip

# 프로젝트 파일 복사 및 설치
WORKDIR /workspace
COPY . /workspace
RUN pip install -e .


# Python Path 설정
ENV PYTHONPATH=$PYTHONPATH:/workspace

# 컨테이너 실행 시 REFace 스크립트 실행
CMD ["sh", "inference_selected.sh"]
