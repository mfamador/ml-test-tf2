import falcon
from falcon import testing
from server import Server


class FalconTestCase(testing.TestCase):
    def setUp(self):
        super(FalconTestCase, self).setUp()
        server = Server()
        self.app = server.create()


class ServerTest(FalconTestCase):
    def testGet(self):
        response = self.simulate_get('/predict/', query_string='retailer=Boots UK')

        print(response.text)
        self.assertEquals(falcon.HTTP_OK, response.status)
        self.assertEquals('Boots', response.text)


    def testGetH(self):
        response = self.simulate_get('/predict/', query_string='retailer=Harrods Inc')

        print(response.text)
        self.assertEquals(falcon.HTTP_OK, response.status)
        self.assertEquals('Harrods', response.text)
