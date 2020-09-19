# Machine Learning experiments

## Install requirements

    pip3.6 install -r requirements.txt 

## Jupyter notebooks to analyse the dataset and create the model for the different scenarios:
    
[multi-label-text-classification.ipynb](notebook/multi-label-text-classification.ipynb)

[time-series-prediction-LSTM.ipynb.ipynb](notebook/time-series-prediction-LSTM.ipynb)

## Train and save the model

    ./1-create-model.sh
or

    python3.6 src/modeltrainer/model_trainer.py -i receipt_data.csv 

## Service to predict retailer name from OCR raw data

   Run predictor service

### Run gunicorn server locally:

    PYTHONPATH=src/retailerpredictor gunicorn -c gunicorn.cfg app:app 
    
### Run gunicorn server with docker:

    ./2-build-docker-image.sh
    ./3-start-docker-container.sh

### Test a prediction

    curl 'http://localhost:8080/predict?retailer=boots%20uk'
   
