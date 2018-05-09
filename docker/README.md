## Before beginning.
+ To work in the timecorr repo from Docker, first clone your fork of the timecorr
+  repo to your computer with `git clone https://github.com/<your-username>/timecorr.git`


## Building the Docker Image.

+ Install [Docker](https://www.docker.com/) and [Google Chrome](https://www.google.com/chrome/browser/desktop/index.html)

+ With Docker running, navigate into the timecorr repo, to the "docker" folder and run the line `docker build -t timecorr .`
+ This creates a docker image according to the specifications in Dockerfile

## Building the Docker container and linking the timecorr repo.

+ To launch the docker image (for the first time) run the following line:
`docker run -it -p 9999:9999 --name TC -v <absolute-path-to-your-timecorr-repo>:/timecorr/ timecorr`

+ These flags broken down are:
  - `-it`: starts the container as an interactive process (opens a shell within it)
  - `-p`: opens and links a port in the docker to a port on your computer (9999 for both)
  - `-name`: specifies a name for the container
  - `-v`: bindmounts a volume on your computer to a newly created directory in the container
  - `timecorr` specifies that the container should be build from the timecorr image created earlier

+ Navigate to the "timecorr" director and run `pip install -e .`

## To launch the tutorial notebook/other notebooks from the docker:

+ navigate to the "notebooks" directory and run the following line
`jupyter notebook demo.ipynb --port=9999 --no-browser --ip=0.0.0.0 --allow-root`
+ and copy/paste the token that is returned into your browser


## After setting up

+ `exit` will exit the bash shell for the docker container
+ `docker stop TC` will stop the container from running
+ `docker start TC` will start he container again (but not bring it up in terminal)
+ `docker exec -it TC bash` will bring up the interactive bash shell for the container

+ if you make a mistake, `docker rm TC` will remove the container so you can reinitialize it
