import sys
import unittest
import os

from test.server_test import ServerTest

if __name__ == '__main__':

    suite = unittest.TestSuite()

    os.environ["RESOURCE_PATH"] = "../../resources"
    os.environ["TERM_SAMPLE_RESOURCE_PATH"] = "src/retailerredictor/test/resources"

    suite.addTest(unittest.makeSuite(ServerTest))
    res = not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful()

    sys.exit(res)
