import logging

import falcon
from retailer_predictor_controller import RetailerPredictorController


class Server():
    def create(self):
        app = falcon.API()
        predictor = RetailerPredictorController()

        app.add_route('/predict', predictor)

        logging.info("Created RetailerPredictorController to be available at /predict")
        return app
