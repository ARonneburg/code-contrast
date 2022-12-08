# Inference server

This module implements simple server to work with **Codify** plugin.
Please visit https://codify.smallcloud.ai for more info about this beautiful tool.

### Usage

Install package and run the server:
```bash
python -m pip install git+https://github.com/smallcloudai/code-contrast.git
python -m api_server.server --workdir /your/working/dir
```

Go to codify settings and set up local inference. Enjoy!

### Run with docker

First you need to install docker. Please follow instruction below:
```shell
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
```
After docker installation and setup pull latest pre-builded image:
```shell
sudo docker pull mityasmc/smc:latest
```
If you have NVIDIA GPU and want to use it, also install nvidia-driver.
Driver tested with image listed below:
```bash
wget https://download.nvidia.com/XFree86/Linux-x86_64/510.85.02/NVIDIA-Linux-x86_64-510.85.02.run
```
Finally start inference server. We recommend use it with GPU:
```shell
# GPU
sudo docker run \
     --gpus 0 \
     -v smc_volume:/working_volume \
     --network="host" \
     mityasmc/smc \
     python -m api_server.run

# CPU
sudo docker run \
     -v smc_volume:/working_volume \
     --network="host" \
     mityasmc/smc \
     python -m api_server.server
```