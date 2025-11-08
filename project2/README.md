# COE 379L Project 2: Damage Classifier

This project demonstrates the training of a model to classify an image as depicting storm damage or not. It packages the best model's weights into an inference server, and packages that inference server into a Docker container. Users may run a local build or use the image published to Dockerhub publicly under my name.

## Server Endpoints

- GET /summary

  Returns information about the model in JSON format

- POST /inference

  Accepts an image binary and returns JSON like

  ```json
  {
    "prediction": "damage"
  }
  ```

  or

  ```json
  {
    "prediction": "no_damage"
  }
  ```

## Inference Server Model Performance

The model being served achieved an F1 of 0.9931972789115646 over the test dataset when it was created.

## Running The Server Build

Note: Only linux/amd64 is supported by the image I published.

Do `docker-compose -f docker-compose.prod.yml up` to start the server.

## Running A Local Build

Do `docker-compose -f docker-compose.local.yml up` to start the server.

## Issuing Commands

By default, the server runs on port 5000. Feel free to change that to whatever works best for you by editing the docker-compose files.

Assuming the server is running on port 5000, the following `curl` examples work.

- GET /summary

  `curl localhost:5000/summary`

- POST /inference

  `curl -X POST -F "file=@/path/to/your/jpg" localhost:5000/inference`
