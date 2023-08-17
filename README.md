# yolov8_API
A simple fastAPI application for a fire detection model. This API should work for any yolov8 onnx model if you just replace the model files and labels

The model files can be found here: https://drive.google.com/drive/folders/1-hp8kwreWYr17R_1OYqmbd4Xl3dFbWHd?usp=sharing 

How to run:

1. First clone the repo and add the correct .onnx model paths
2. Build the docker container : docker build -t my-fastapi-app .
3. Run the docker container : docker run -d -p 8000:8000 --name container_name my-fastapi-app

