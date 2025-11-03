#!/bin/bash
# Script to build the Docker image for the FRS Microservice

IMAGE_NAME="frs-microservice:latest"

echo "--- Building Docker image: $IMAGE_NAME ---"
docker build -t $IMAGE_NAME .

if [ $? -eq 0 ]; then
    echo "--- Docker image built successfully! ---"
    echo "To run the container, use:"
    echo "docker run -d -p 8000:8000 --name frs_app $IMAGE_NAME"
    echo "The API documentation will be available at http://localhost:8000/docs"
else
    echo "--- Docker image build failed. ---"
fi
