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

RUN adduser --disabled-password --gecos '' --ingroup adm --shell /bin/bash --uid 1337 user && chown -R user:adm /home/user
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user

RUN echo "export PATH=~/.local/bin:/usr/local/cuda/bin:\$PATH" > /etc/profile.d/50-smc.sh
ENV PATH=/home/user/.local/bin:$PATH

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

USER user
RUN pip install --no-cache-dir IPython numpy tokenizers fastapi uvicorn termcolor cdifflib cloudpickle dataclasses_json
RUN pip install --no-cache-dir torch --extra-index-url https://download.pytorch.org/whl/cu116
RUN pip install --no-cache-dir git+https://github.com/smallcloudai/code-contrast.git

CMD ["/bin/bash"]