import pandas as pd
import numpy as np

from sklearn import svm 
from sklearn.ensemble import GradientBoostingRegressor
#from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


from utils import Utils

class Models:

    def __init__(self):
        self.classifier = svm.SVC(kernel='linear')


    def model_training(self,X,y):

        self.classifier.fit(X, y)

        utils = Utils()
        utils.model_export(self.classifier)

    def model_prediction(self,X):

        X_prediction = self.classifier.predict(X)
        return X_prediction

    def model_evaluation(self,prediction,y):
        data_accuracy = accuracy_score(prediction,y)
        print("="*70)
        return data_accuracy