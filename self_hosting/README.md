# Inference server

This module implements simple server to work with **Codify** plugin.
Please visit https://codify.smallcloud.ai for more info about this beautiful tool.

## Get started

### Prepare docker

To run server you need NVIDIA GPU with at least 6Gb memory and nvidia docker.
Make sure you have the latest version of nvidia driver installed.
Image tested with nvidia driver of version *510.85.02*.

<details>
<summary> if you don't have nvidia docker </summary>

#### Linux

To install nvidia docker please follow this
[instruction](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

Next add your user to docker group (to run docker without sudo):
```bash
sudo usermod -aG docker <your user>
```

#### Windows 10, 11

Please follow this [guide](https://docs.docker.com/desktop/install/windows-install).
Note that docker needs WSL2.

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
https://localhost:8008
```

Now you can use your own server for codify!