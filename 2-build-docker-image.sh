#!/bin/bash

docker rm -f retailer-predictor
docker rmi -f retailer-predictor
docker image build --tag retailer-predictor .
