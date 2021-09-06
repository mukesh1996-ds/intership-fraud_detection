import logging as lg
import pandas as pd

lg.basicConfig(filename="class.log", level = lg.DEBUG, format="%(asctime)s, %(lineno)s,%(message)s")
lg.info("This class file is used to check is their any null value and type of data")
class Check_model:
    def __init__(self, data):
        try:
            self.data = data
        except Exception as e:
            lg.error(e)

    def is_null(self):
        lg.info("checking for null value")
        try:
            print(self)
        except Exception as e:
            lg.error(e)
    def data_type(self):
        lg.info("checking the data type of the value")
        try:
            print(type(self))
        except Exception as e:
            lg.error(e)



