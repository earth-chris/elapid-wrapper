#!/bin/bash

echo "preparing docker image for pipeline"
echo "building docker image"
docker build  --tag earthchris/ccb-pytest:latest .
echo "pushing docker image to public repository"
docker push earthchris/ccb-pytest
