import logging

import falcon
import model_controller


class RetailerPredictorController(object):

    def __init__(self):
        self.model = model_controller.load_model()

    def on_get(self, req, resp):
        resp.status = falcon.HTTP_200
        raw_retailer = req.get_param('retailer', '')
        logging.info("Received request retailer={} ".format(raw_retailer))
        print("Received request retailer={} ".format(raw_retailer))

        if len(raw_retailer) > 1:
            retailer_prediction = self.predict_retailer(raw_retailer)
            resp.status = falcon.HTTP_200
        else:
            retailer_prediction = 'you have to specify the retailer'
            resp.status = falcon.HTTP_400
        resp.body = retailer_prediction

    def predict_retailer(self, feature):
        try:
            prediction = self.model.predict(feature)

            return prediction
        except Exception:
            logging.exception("Error predicting retailer for request.")
            raise falcon.HTTPInternalServerError(description='Error predicting retailer')
