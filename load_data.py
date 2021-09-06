import pandas as pd
import logging as lg
lg.basicConfig(filename="load_data.log", level=lg.DEBUG, format="%(asctime)s, %(message)s")

def load_csv(filepath):
    try:
        data =  []
        col = []
        checkcol = False
        with open(filepath) as f:
            for val in f.readlines():
                val = val.replace("\n","")
                val = val.split(',')
                if checkcol is False:
                    col = val
                    checkcol = True
                else:
                    data.append(val)
        df = pd.DataFrame(data=data, columns=col)
        return df
    except Exception as e:
        lg.error(e)

