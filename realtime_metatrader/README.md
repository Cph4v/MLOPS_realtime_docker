
## ---> chown root  /config/.wine

docker image ls
docker volume ls
docker container rm e8f82d4af439
docker container stop e8f82d4af439
docker rmi -f  fe0e5e9d92a7

docker rmi -f  mt5
docker container stop e8f82d4af439
docker exec -it 87cd17088171 bash
echo "$USER"


!! DANGER - delete all stope containers
docker container prune
docker volume prune
docker volume ls 
docker volume rm config
docker volume create vscode-dev-1


## To delete all containers including its volumes use,
 - docker rm -vf $(docker ps -aq)
## To delete all the images,
 - docker rmi -f $(docker images -aq)


docker build --no-cache -t mt5 .
sudo docker run -d -p 3000:3000 -p 8001:8001 -v config:/config mt5
sudo docker run -d -p 3000:3000 -p 8001:8001 mt5


## docker 

 - docker compose --progress plain build --no-cache  2>&1 | tee build.log
 - docker compose --progress plain build 2>&1 | tee build.log
 - docker compose up
 - docker compose up -d
 - docker compose down
 - docker compose rm

chown -R root /config/.wine






# MetaTrader5 Docker Image

This project provides a Docker image for running MetaTrader5 with remote access via VNC, based on the [KasmVNC](https://github.com/kasmtech/KasmVNC) project and [KasmVNC Base Image from LinuxServer](https://github.com/linuxserver/docker-baseimage-kasmvnc).

## Features

- Run MetaTrader5 in an isolated environment.
- Remote access to MetaTrader5 interface via an integrated VNC client accessible through a web browser.
- Built on the reliable and secure [KasmVNC](https://github.com/kasmtech/KasmVNC) project.
- RPyC server for remote access to Python MetaTrader Library from Windows or Linux using https://github.com/lucas-campagna/mt5linux

![MetaTrader5 running inside container and controlled through web browser](https://imgur.com/v6Hm9pa.png)

----------

**NOTICE:**
Due to some compatibility issued, version 2 has switched its base from Alpine to Debian Linux. This and adding Python environment makes that container size is considerably bigger from about 600 MB to 4 GB.

If you just need to run Metatrader for running your MQL5 programs without any Python programming I recommend to go on using version 1.0. MetaTrader program is updated independently from image so you will always have latest MT5 version.

-----------

## Requirements

- Docker installed on your machine.
- Only intelx86/amd64 host is supported

## Usage from repository

1. Clone this repository:
```bash
git clone https://github.com/gmag11/MetaTrader5-Docker-Image
cd MetaTrader5-Docker-Image
```

2. Build the Docker image:
```bash
docker build -t mt5 .
```

3. Run the Docker image:
```bash
docker run -d -p 3000:3000 -p 8001:8001 -v config:/config mt5
```

Now you can access MetaTrader5 via a web browser at localhost:3000.

On first run it may take a few minutes to get everything installed and running. Normally it takes less than 5 minutes. You don't need to do anything. All installation process is automatic and you should end up with MetaTrader5 running in your web session.

## Usage with docker compose with image form Docker Registry (preferred way)

1. Create a folder in a path where you have permission. For instance in your home.
```bash
mkdir MT5
cd MT5
```

2. Create `docker-compose.yaml` file.
```bash
nano docker-compose.yaml
```

Use this content filling user and password with your own data.

```yaml
version: '3'

services:
  mt5:
    image: gmag11/metatrader5_vnc
    container_name: mt5
    volumes:
      - ./config:/config
    ports:
      - 3000:3000
      - 8001:8001
    environment:
      - CUSTOM_USER=<Choose a user>
      - PASSWORD=<Choose a secure password>
```

**Notice**: If you do not need to do remote python programming you can get a much smaller installation changing this line:

```yaml
image: gmag11/metatrader5_vnc
```

by this one


```yaml
image: gmag11/metatrader5_vnc:1.0
```

3. Start the container
```bash
docker compose up -d
```

In some systems `docker compose` command does not exists. Try to use `docker-compose up -d` instead.

4. Connect to web interface
   Start your browser pointing http://<your ip address>:3000

On first run it may take a few minutes to get everything installed and running. Normally it takes less than 5 minutes. You don't need to do anything. All installation process is automatic and you should end up with MetaTrader5 running in your web session.

## Where to place MQ5 and EX5 files
In the case you want to run your own MQL5 bots inside the container you can find MQL5 folder structure in 

```
config/.wine/drive_c/Program Files/MetaTrader 5/MQL5
```

All files that you place there can be accessed from your MetaTrader container without the need to restart anything.

You can access MetaEditor program clicking in `IDE` button in MetaTrader5 interface.

**Notice**: If you will run MQL5 only bots (without Python) you can run perfectly with gmag11/metatrader5_vnc:1.0 image as pointed before. Remember that **image version is not stuck to a specific MetaTrader 5 version**.

**Metatrader will always be updated automatically to latest version as it does when it is nativelly installed in Windows.**

## Python programming

You need to install [mt5linux library](https://github.com/lucas-campagna/mt5linux) in your Python host. It may be in any OS, not only Linux.

This is a simple snippet to run your Python script fron any host

```python
from mt5linux import MetaTrader5
mt5 = MetaTrader5(host='host running docker container',port=8001)
mt5.initialize()
print(mt5.version())
```

Output should be something like this:

```
(mt5linux) linux:~/$ python3
Python 3.10.13 (main, Dec 26 2023, 20:21:41) [GCC 13.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> from mt5linux import MetaTrader5
>>> mt5 = MetaTrader5(host='192.168.1.10',port=8001)
>>> mt5.initialize()
True
>>> print(mt5.version())
(500, 4120, '22 Dec 2023')
>>>
```
