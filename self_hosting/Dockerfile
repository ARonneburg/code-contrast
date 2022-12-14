FROM nvidia/cuda:11.6.1-cudnn8-devel-ubuntu20.04

RUN apt-get update
RUN DEBIAN_FRONTEND="noninteractive" apt-get install -y software-properties-common wget

RUN apt-get update
RUN DEBIAN_FRONTEND="noninteractive" apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    git-lfs \
    openssh-server \
    rsyslog \
    htop \
    tmux \
    vim \
    rsync \
    inetutils-ping net-tools psmisc telnet \
    p7zip \
    python3 python3-pip \
    autotools-dev automake build-essential cmake \
    clang-11 lldb-11 lld-11 clangd-11 \
    pkg-config \
    lsof \
    kmod \
    udev \
    debhelper \
    libncurses5-dev libncursesw5-dev \
    && rm -rf /var/lib/{apt,dpkg,cache,log}

RUN echo "export PATH=~/.local/bin:/usr/local/cuda/bin:\$PATH" > /etc/profile.d/50-smc.sh
ENV PATH=/home/user/.local/bin:$PATH

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

RUN pip install --no-cache-dir IPython numpy tokenizers fastapi uvicorn termcolor cdifflib
RUN pip install --no-cache-dir cloudpickle dataclasses_json huggingface_hub blobfile
RUN pip install --no-cache-dir torch --extra-index-url https://download.pytorch.org/whl/cu116
RUN pip install --no-cache-dir git+https://github.com/smallcloudai/code-contrast.git

RUN mkdir /workdir
ENV SERVER_WORKDIR=/workdir
ENV SERVER_PORT=8008

CMD ["python", "-m", "self_hosting.watchdog", "--workdir", "/workdir"]