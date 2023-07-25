FROM hccz95/ubuntu:18.04-py38

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        build-essential freeglut3-dev libglib2.0-dev libxrender-dev fontconfig && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# gym==0.21.0 need setuptools==63.2.0
RUN pip install --no-cache-dir \
        setuptools==63.2.0
RUN pip install --no-cache-dir \
        numpy==1.19.3 \
        pygame==2.1.2 pyglet==1.5.21 \
        scipy torch==1.9.0 \
        gym==0.21.0 \
        opencv-contrib-python==4.2.0.34

RUN git clone https://github.com/openai/multiagent-particle-envs.git && \
    cd multiagent-particle-envs && \
    sed -i 's/random_array = prng.np_random.rand(self.num_discrete_space)/random_array = np.random.RandomState().rand(self.num_discrete_space)/g' multiagent/multi_discrete.py && \
    sed -i 's/from gym.spaces import prng/# from gym.spaces import prng/g' multiagent/multi_discrete.py && \
    pip install -e .
ENV SUPPRESS_MA_PROMPT=1

# install marllib
RUN pip install --no-cache-dir setuptools==63.2.0
RUN git clone https://github.com/Replicable-MARL/MARLlib.git && \
    cd MARLlib && pip install --no-cache-dir -e . && \
    pip install --no-cache-dir "gym>=0.20.0,<0.22.0" protobuf==3.20.0 pyglet==1.5.11 && \
    cd marllib/patch && \
    python add_patch.py -y
