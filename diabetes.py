import numpy as np
import pandas as pd

from utils import Utils
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

if __name__ == "__main__":

    #######################################################
    #Data collection and analysis

    #loading dataset
    diabetes_ds = pd.read_csv('./in/diabetes.csv')
    print(diabetes_ds.head())

    #number of rows and columns

    print("="*70)
    print("Number of rows and columns: ",diabetes_ds.shape)

    #getting the statistical measures
    print("="*70)
    print(diabetes_ds.describe())

    print("="*70)
    print(diabetes_ds['Outcome'].value_counts())

    """
    0-->non-Diabetic
    1-->Diabetic
    """
    print("="*70)
    print(diabetes_ds.groupby('Outcome').mean())

    #separating the data and labels
    X = diabetes_ds.drop(columns='Outcome',axis=1)
    y = diabetes_ds['Outcome']

    #DAta standarization--the range are diferent

    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    print("="*70)
    print(X)

    #features stract

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, stratify=y, random_state=42)
    print(X.shape, X_train.shape, X_test.shape)

    classifier = svm.SVC(kernel='linear')
    #training the suppoort vector machinme Classifier
    classifier.fit(X_train,y_train)

    #Model ecaluation
    #Accuracy score

    #accuracy score on the trainin data
    X_train_prediction = classifier.predict(X_train)
    training_data_accuracy = accuracy_score(X_train_prediction,y_train)

    print("="*70)

    print('Accuracy score of the training data: ', training_data_accuracy)

    #accuracy score on the test data
    X_test_prediction = classifier.predict(X_test)
    test_data_accuracy = accuracy_score(X_test_prediction,y_test)
    print("="*70)
    print('Accuracy score of the test data: ', test_data_accuracy)

    #Making a predictive system

    input_data = (0,137,40,35,168,43.1,2.288,33)

    #changingn the input data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    #reshape the array as we are predicting for one instance
    input_data_reshaped= input_data_as_numpy_array.reshape(1,-1)

    # standarize input data
    std_data = scaler.transform(input_data_reshaped)
    print(std_data)

    prediction = classifier.predict(std_data)
    print(prediction)

    if prediction[0] == 0:
        print('The person is not diabetic')
    else:
        print('The person is diabetic')

 