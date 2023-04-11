<div align="center">

# <font color="red">[{</font> Refact.ai Inference Server
### Self-hosted server for [refact.ai](https://www.refact.ai) coding assistant.

</div>

## Demo

<table align="center">
<tr>
<th><img src="https://plugins.jetbrains.com/files/20647/screenshot_277b57c5-2104-4ca8-9efc-1a63b8cb330f" align="center"/></th>
<th><img src="https://plugins.jetbrains.com/files/20647/screenshot_a57c91d1-b841-495e-9e81-9af2129bef24" align="center"/></th>
</tr>
</table>

## Features
* Plugins for [JetBrains](https://plugins.jetbrains.com/plugin/20647-refact-ai) products and
  [VSCode IDE](https://marketplace.visualstudio.com/items?itemName=smallcloud.codify)
* Multilingual [models](https://huggingface.co/smallcloudai) under the hood. 20+ languages support
  (python, java, c++, php, javascript, go and others)
* Self-hosted server running with Docker
* Completion and AI Toolbox
* Privacy settings for projects or even single files

TODO
<p align="center">
<img src="https://www.refact.ai/images/scheme.svg" style="background-color:white;padding:20px;">
</p>

Join our [Discord server](https://www.smallcloud.ai/discord) and follow our
[Twitter](https://twitter.com/refact_ai) to get the latest updates.

## Contents
* [Prerequisities](#prerequisities)
* [Getting started](#getting-started)
* [Plugins usage](#plugins-usage)
* [Contributing](#contributing)

## Prerequisities
- Install plugin for your IDE: [JetBrains](https://plugins.jetbrains.com/plugin/20647-refact-ai) or
  [VSCode](https://marketplace.visualstudio.com/items?itemName=smallcloud.codify)
- TODO: Login and choose SELF-HOSTED plan for your account
- Large Language Models require a lot of computing resources and memory.
  We strongly recommend use this server with **Nvidia GPU with at least 4Gb VRAM**.
  Another option is to use it with CPU, but it's quite slow and unusable in practice yet.

## Getting started

Recommended way to run server is pre-builded Docker image.

### Docker
Install [Docker with NVidia GPU support](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker). Then get **API Key** from refact.ai [account](https://codify.smallcloud.ai/account)
page or alternatively from plugin settings.

<details><summary>Docker tips & tricks</summary>

Add your yourself to docker group to run docker without sudo (works for Linux):
```commandline
sudo usermod -aG docker {your user}
```
List all containers
```commandline
docker ps -a
```
Create a new container
```commandline
docker run
```
Start and stop existing containers, but it doesn't remove them
```commandline
docker start
docker stop
```
Remove a container and all its data
```commandline
docker rm
```
Model weights are saved inside the container. If you remove the container, it will
download the weights again.
</details>

Run docker container with following command:
```commandline
docker run --gpus 0 --name refact_self_hosting -p 8008:8008 --env SERVER_API_TOKEN={API Key} smallcloud/self_hosting
```
Next time you can start it with following command:
```commandline
docker start -i refact_self_hosting
```
After start, container will automatically check for updates and download the chosen model
(see in your [account](https://codify.smallcloud.ai/account)).

### Manual installation

Alternative way to run server is manual package installation:
```commandline
pip install git+https://github.com/smallcloudai/code-contrast.git
```
You can run server with following command:
```commandline
python -m self_hosting.server --workdir /workdir --token {API Key}
```

## Plugins usage

Go to plugin settings and set up custom inverence url:
```commandline
https://localhost:8008
```
<details><summary>JetBrains</summary>
Settings > Tools > Refact.ai > Advanced > Inference URL
</details>
<details><summary>VSCode</summary>
Extensions > Refact.ai Assistant > Settings > Infurl
</details>

Make sure your server started with same API Key.

Now it should work, just try to write some code! If it doesn't, please report your experience to
[GitHub issues](https://github.com/smallcloudai/code-contrast/issues).

## Contributing

TODO