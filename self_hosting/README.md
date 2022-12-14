# Inference server

This module implements simple server to work with **Codify** plugin.
Please visit https://codify.smallcloud.ai for more info about this beautiful tool.

### Run with docker

First you need to install docker. Please follow instruction below:
```shell
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
```
After docker installation and setup pull latest pre-builded image:
```shell
sudo docker pull smallcloud/self_hosting:latest
```
Make sure that you have nvidia driver and GPU with at least 6Gb. Driver tested with image listed below:
```bash
wget https://download.nvidia.com/XFree86/Linux-x86_64/510.85.02/NVIDIA-Linux-x86_64-510.85.02.run
```
Get codify api key from plugin and start inference server:
```shell
sudo docker run \
     --gpus 0 \
     -v self_hosting:/workdir \
     --network="host" \
     --env SERVER_API_TOKEN=<your plugin key> \
     smallcloud/self_hosting
```
Go to plugin settings and set custom inference url:
```shell
http://localhost:8008  # if you run server locally
http://server_host_name:8008  # otherwise
```