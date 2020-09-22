# Machine Learning experiments with Tensorflow 2

## Jupyter notebooks to analyse the dataset and create the model for the different scenarios:

    docker run --name jupyter --rm -v $PWD:/home/jovyan -p 8888:8888 jupyter/tensorflow-notebook

navigate to `notebooks` and open 

    [multi-label-text-classification.ipynb](notebook/multi-label-text-classification.ipynb)

    [time-series-prediction-LSTM.ipynb.ipynb](notebook/time-series-prediction-LSTM.ipynb)


# Dockerize and run a service to make predictions

## Install requirements

    pip3 install -r requirements.txt 


## Train and save the model

    ./1-create-model.sh
or

    python3 src/modeltrainer/model_trainer.py -i receipt_data.csv 

## Service to predict retailer name from OCR raw data

   Run predictor service

### Run gunicorn server locally:

    PYTHONPATH=src/retailerpredictor gunicorn -c gunicorn.cfg app:app 
    
### Run gunicorn server with docker:

    ./2-build-docker-image.sh
    ./3-start-docker-container.sh

### Test a prediction

    curl 'http://localhost:8080/predict?retailer=boots%20uk'
   
