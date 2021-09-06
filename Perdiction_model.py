"""As for the flow of articture we also need to preform all the data preprocessing steps"""
"""Loading all the required packages"""
import pandas as pd
import matplotlib.pyplot as plt
import logging as lg
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from check_data import Check_model
from load_data import load_csv
import pickle

# create a pickle file
lg.basicConfig(filename="Prediction_model.log", level = lg.DEBUG, format="%(asctime)s, %(lineno)s, %(message)s")

lg.info("checking for data loading into training_data")

try:
    predicting_data = load_csv("F:\InternShip\Fraud Detection\Training_model\input_data.csv")
    print(predicting_data.head())
except Exception as e:
    lg.error(e)

"""Creating the function for data checking null values"""
"""Creating
 the function for data checking null values"""

lg.info("checking for null values in dataset")
null = Check_model(predicting_data)
print(null.is_null())
lg.info("Their is none null value in the data set")
lg.info("checking for type of data in  dataset")
type_data = Check_model(predicting_data)
print(type_data.data_type())
lg.info("Checking the type of data is done successfully")


"""spliting the input data into two different parts"""

x = predicting_data.drop(['fraud'],axis=1)
y = predicting_data['fraud']
print(x.head())
print(y.head())

"""over sampling """
lg.info("Using SMOTE(Synthetic Minority Oversampling Technique) [2] for balancing the dataset.")

try:
    sm = SMOTE(random_state=42)
    x_res,y_res = sm.fit_resample(x, y)
    y_res = pd.DataFrame(y_res)
    print(y_res[0].value_counts())
except Exception as e:
    lg.error(e)

"""Train Test split"""
lg.info("Beging of Train Test Split")
try:
    x_train, x_test, y_train, y_test = train_test_split(x_res,y_res,test_size=0.3,random_state=42, shuffle=True, stratify=y_res)
    print(x_train.shape)
    print(x_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    lg.info("Success of train test split")
except Exception as e:
    lg.error(e)
# function for ploting AUC-CURVE
def plot_roc_auc(y_test, preds):
    '''
    Takes actual and predicted(probabilities) as input and plots the Receiver
    Operating Characteristic (ROC) curve
    '''
    try:
        fpr, tpr, threshold = roc_curve(y_test, preds)
        roc_auc = auc(fpr, tpr)
        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()
        lg.warning("DataConversionWarning")
    except Exception as e:
        lg.error(e)

lg.info("K-Neighbours Classifier")
knn = KNeighborsClassifier(n_neighbors=5,p=1)
knn.fit(x_train,y_train)
y_pred = knn.predict(x_test)
print("Classification Report for K-Nearest Neighbours: \n", classification_report(y_test, y_pred))
print("Confusion Matrix of K-Nearest Neigbours: \n", confusion_matrix(y_test,y_pred))
plot_roc_auc(y_test, knn.predict_proba(x_test)[:,1])
lg.warning("DataConversionWarning")
lg.info("K-Neighbours Classifier completed with good accuracy score")

lg.info("Random Forest Classifier")
rf_clf = RandomForestClassifier(n_estimators=100,max_depth=8,random_state=42,
                                verbose=1,class_weight="balanced")
rf_clf.fit(x_train,y_train)
y_pred = rf_clf.predict(x_test)

print("Classification Report for Random Forest Classifier: \n", classification_report(y_test, y_pred))
print("Confusion Matrix of Random Forest Classifier: \n", confusion_matrix(y_test,y_pred))
plot_roc_auc(y_test, rf_clf.predict_proba(x_test)[:,1])
lg.info("Random ForestClassifier completed with good excelent score")

"""Dumping the file in pickle format"""
pickle.dump(rf_clf, open('model.pkl', 'wb'))
model = pickle.load(open('model.pkl', 'rb'))



