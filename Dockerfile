FROM hccz95/ubuntu:20.04-py38

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        build-essential freeglut3-dev libglib2.0-dev libxrender-dev fontconfig && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# install dependencies for fortattack
# gym==0.21.0 need setuptools==63.2.0
RUN pip install --no-cache-dir \
        setuptools==63.2.0
RUN pip install --no-cache-dir \
        numpy==1.19.3 \
        pygame==2.1.2 pyglet==1.5.21 \
        scipy torch==1.9.0 \
        gym==0.21.0 gymnasium \
        opencv-contrib-python==4.2.0.34

# install dependencies for marllib
RUN pip install --no-cache-dir setuptools==63.2.0
RUN git clone https://gitee.com/hccz95/MARLlib.git && \
    cd MARLlib && \
    pip install --no-cache-dir --upgrade-strategy only-if-needed -e . && \
    cd marllib/patch && \
    python add_patch.py -y
