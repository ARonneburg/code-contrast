# Inference server

This module implements simple server to work with **Codify** plugin.
Please visit https://codify.smallcloud.ai for more info about this beautiful tool.

## Get started

### Prepare docker

To run server you need NVIDIA GPU with at least 6Gb memory and nvidia docker.

<details>
<summary> if you don't have nvidia docker </summary>

First you need to install nvidia driver to your system. We tested image with driver listed below:
```bash
wget https://download.nvidia.com/XFree86/Linux-x86_64/510.85.02/NVIDIA-Linux-x86_64-510.85.02.run
```

To install nvidia docker please follow instruction below:
```shell
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
```

Add your user to docker group to run docker without sudo:
```bash
sudo usermod -aG docker <your user>
```

</details>

### Run server

Get codify api key from plugin and run inference container:
```shell
docker run --gpus 0 --name self_hosting -p 8008:8008 --env SERVER_API_TOKEN=<token> smallcloud/self_hosting
```

<details>
<summary> note </summary>

Next time you can start in with following command:
```shell
docker start -i self_hosting
```

</details>

Go to plugin settings and set custom inference url:
```shell
http://localhost:8008  # if you run server locally
http://server_host_name:8008  # otherwise
```

Now you can use your own server for codify!