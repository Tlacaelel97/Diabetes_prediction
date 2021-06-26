# Diabetes prediction

Model that helps to predict if a pacient has or not diabetes

The model use a dataset from pacients that includes Pregnancies, Glucose, Blood Pressure, Skin Thickness, 
Insulin, BMI, Diabetes Pedigree Function, Age, Outcome, where Outcome represents if the pacient is diabetic (1)
or non-diabetic (0).

# Models

The [models.py](models.py) includes the following classes:

• [__init__](__init__) - Define the machine learning model.

• model_training - Train the model and export in a pickle archive

• model_prediction - prediction 

• model_evaluation - Evaluate the accuracy of the model

# Utils

The [utils.py](utils.py) includes the following classes:

load_from_csv(self,path):
        return pd.read_csv(path)

    def load_from_mysql(self):
        pass

    def show_info(self,data):
        print(data.head())
        #number of rows and columns
        print("="*70)
        print("Number of rows and columns: ",data.shape)

    #statistical measures
    def stat_measures(self,data):
        print("="*70)
        print(data.describe())

    #output values
    def out_measures(self,data):
        print("="*70)
        print("Number of output values, where 0 = non-diabetic and 1 = diabetic")
        print(data['Outcome'].value_counts())
        print("="*70)
        print(data.groupby('Outcome').mean())

    def features_target(self,dataset,drop_cols, y):
        X=dataset.drop(drop_cols, axis=1)
        y = dataset[y]
        return X,y

    def features_extract(self,X,y):
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, stratify=y, random_state=42)
        print(X.shape, X_train.shape, X_test.shape)
        return X_train, X_test, y_train, y_test

    def model_export

run the [main.py](main.py) to train the model and test with one data


