## Show docker disk usage
- df -H
- docker system df
- clean docker : https://depot.dev/blog/docker-clear-cache

## docker 
- docker compose --progress plain build --no-cache  2>&1 | tee build.log
- docker compose --progress plain build 2>&1 | tee build.log
- docker compose up
- docker compose up -d
- docker compose down
- docker compose rm
- docker compose up --build


## Docker commands:
- docker image ls
- docker volume ls
- docker network ls
- docker container rm 
- docker container stop 
- docker rmi -f  

- docker rmi -f  
- docker container stop 
- docker exec -it  bash
- echo "$USER"


## !! DANGER - delete all stope containers
- docker container prune
- docker volume prune
- docker volume ls 
- docker volume rm config
- docker volume create vscode-dev-1


## To delete all containers including its volumes use,
- docker rm -vf $(docker ps -aq)
## To delete all the images,
- docker rmi -f $(docker images -aq)


## Removing docker build cache
- docker builder prune