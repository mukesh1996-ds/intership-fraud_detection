"""Loading all the required packages"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging as lg
from check_data import Check_model
from load_data import load_csv

lg.basicConfig(filename="Training_model.log", level = lg.DEBUG, format="%(asctime)s, %(lineno)s, %(message)s")
lg.info("checking for data loading into training_data")
try:
    training_data = load_csv("F:\InternShip\Fraud Detection\Fraud_detection.csv")
    print(training_data.head())
except Exception as e:
    lg.error(e)
lg.info("checking for null values in dataset")
try:
    null = Check_model(training_data)
    print(null.is_null())
    lg.info("Their is none null value in the data set")
    lg.info("checking for type of data in  dataset")
    type_data = Check_model(training_data)
    print(type_data.data_type())
    lg.info("Checking the type of data is done successfully")
except Exception as e:
    lg.error(e)

# data seperating fraud and non fraud
try:
    lg.info("creating two data frame name fraud and non_fraud")
    lg.info("fraud data")
    fraud = training_data.loc[training_data.fraud == 1]
    print(fraud.head())
    lg.info("Non_fraud data")
    non_fraud = training_data.loc[training_data.fraud == 0]
    print(non_fraud.head())
except Exception as e:
    lg.error(e)

# visualizing
lg.info("checking the difference btw fraud and non_fraud by visualizing it ")
try:
    sns.countplot(x = 'fraud', data = training_data)
    plt.title("Count of fraud payment by customer")
    plt.show()
    lg.info("visualization done")
except Exception as e:
    lg.error(e)

# checking the count of fraud and non fraud
lg.info("counting the number of fraud and non fraud ")
try:
    print("Number of non fraud people are:", non_fraud.fraud.count())
    print("Number of fraud people are:", fraud.fraud.count())
except Exception as e:
    lg.error(e)
# people fraud on which category more
lg.info("Checking the mean of amount and fraud")
try:
    print("Mean feature values per category",training_data.groupby('category')['amount','fraud'].mean())
except Exception as e:
    lg.error(e)

lg.info("DATA PREPROCESSING")
""" Deleting the zipCodeOri and zipMerchant as their are only single value in the dataset """
lg.info("checking for zipCodeOri data counts")
try:
    print("Unique zipCodeOri values: ",training_data.zipcodeOri.nunique())
    lg.info("checking for zipCodeOri data counts")
    print("Unique zipMerchant values: ",training_data.zipMerchant.nunique())
    lg.info('Droping successful ')
    data_reduced = training_data.drop(['zipcodeOri','zipMerchant'],axis=1, inplace=True)
    lg.info("checking the name of the columns Done")
    print(training_data.columns)
except Exception as e:
    lg.error(e)


"""Converting cartigorical"""
lg.info("Converting the data from cateogrical ")
col_categorical = training_data.select_dtypes(include= ['object']).columns
print(col_categorical)
lg.info("Conversion started and iteration started")
for col in col_categorical:
    try:
        lg.info("iteration on all the data in dataset")
        training_data[col] = training_data[col].astype('category')
    except Exception as e:
        lg.error(e)
# conversion
try:
    lg.info("Conversion done")
    training_data[col_categorical] = training_data[col_categorical].apply(lambda x: x.cat.codes)
    print("Final data is ready: ", training_data.head())
except Exception as e:
    lg.error(e)

"""Saving the dataset """
lg.info("SAVING THE DATA SET AS INPUT.CSV WHICH WILL BE APPLIED TO PREDICTION_MODEL FOR PREDICTION PURPOSE")
training_data.to_csv('input_data.csv',index = False )
lg.info("EVERY THING DONE AND DATA SET IS SAVED")
    
