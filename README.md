# Deploy a web service for YOLOv5 using Flask

The purpose of this tutorial is to show how to deploy a web service for YOLOv5 using your own weights generated after training a YOLOv5 model on your dataset.


## Instructions before using code

First, the tree structure of your folder should be as follows.

![image](tree_yolov5_web_service.png)

First, you have to create the folder named `models_train` and this is where you can store the weights generated after your trainings. You are free to put as many weight files as you want in the `models_train` folder.

**Then you can use the code above !**


## Test it locally (Optional)

Launch the following **docker command** or local virtual environment to launch your application locally on your computer:

```console

docker build . -t yolov5_web:latest
docker run --rm -it -p 5000:5000 yolov5_web:latest
```
```console
pip install -r torch_requirements.txt
pip install -r requirements.txt
python app.py
```

> :heavy_exclamation_mark: The `-p 5000:5000` argument indicates that you want to execute a port rediction from the port **5000** of your local machine into the port **5000** of the docker container. The port **5000** is the default port used by **Flask** applications.
>


Once started, your application should be available on http://localhost:5000.


Credit:
This repo is modified from https://github.com/ovh/ai-training-examples/tree/main/jobs/yolov5-web-service
