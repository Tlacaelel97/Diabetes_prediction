from utils import Utils
from models import Models
import numpy as np

from sklearn.preprocessing import StandardScaler

if __name__ == "__main__":

    utils = Utils()
    models=Models()

    #loading dataset
    diabetes_ds = utils.load_from_csv('./in/diabetes.csv')
    
    #Show data
    utils.show_info(diabetes_ds)

    #getting the statistical measures
    utils.stat_measures(diabetes_ds)

    #getting output measures
    utils.out_measures(diabetes_ds)

    #separating the data and labels

    X,y = utils.features_target(diabetes_ds,['Outcome'],['Outcome'])

    #DAta standarization--the range are diferent

    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    print("="*70)

    #features stract

    X_train, X_test, y_train, y_test = utils.features_extract(X,y.values.ravel())


    #training the suppoort vector machinme Classifier
    models.model_training(X_train,y_train)

    #Model ecaluation
    #Accuracy score 

    #accuracy score on the trainin data
    X_train_prediction = models.model_prediction(X_train)
    training_data_accuracy = models.model_evaluation(X_train_prediction,y_train)
    print('Accuracy score of the training data: ', training_data_accuracy)

    #accuracy score on the test data
    X_test_prediction = models.model_prediction(X_test)
    test_data_accuracy = models.model_evaluation(X_test_prediction,y_test)
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

    prediction = models.model_prediction(std_data)
    print(prediction)

    if prediction[0] == 0:
        print('The person is not diabetic')
    else:
        print('The person is diabetic')

