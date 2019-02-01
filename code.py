#Importing the libs and dataset
import numpy as np
import matplotlib.pyplot as plt, matplotlib.image as mpimg
import pandas as pd
import sklearn
from joblib import dump,load
import xgboost as xgb
from xgboost.sklearn import XGBClassifier



train  = pd.read_csv('train.csv')
test  = pd.read_csv('test.csv')

#timages = test.iloc[0:1000,:]
images = train.iloc[0:1000,1:]
labels = train.iloc[0:1000,0]


#Splitting the training set
from sklearn.model_selection import train_test_split
images_train, images_test, labels_train, labels_test = train_test_split(images, labels, test_size = 0.2, random_state = 0)

#Checking for missing values
images_train.isnull().any().describe()
labels_train.isnull().any().describe()

#from xgboost import XGBClassifier
classifier = XGBClassifier(silent = 0, eta = 0.1, max_depth = 8, subsample = 0.75, colsample_bytree = 0.75)
classifier.fit(images, labels)


# Predicting the Test set results
y_pred = classifier.predict(test)

#Checking score (Accuracy)
classifier.score(images_test,labels_test)

#Creating the joblib file
dump(classifier, 'xgb.joblib')

"""#Getting the joblib file
classifier = load('random_forest.joblib')"""
#Generating the final dataframe
y_pred = pd.DataFrame(y_pred)
y_pred[:,1] = y_pred[:,0]
y_pred['ImageId'] = pd.Series(data = np.arange(1,28001), index=y_pred.index)
y_pred.columns = ['Label','ImageId']
#y_pred = y_pred.drop(columns = ['ImageId'])
columnsTitles=["ImageId","Label"]
y_pred=y_pred.reindex(columns=columnsTitles)

#Exporting the dataframe
y_pred.to_csv('Predictions.csv')



# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels_test, y_pred)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = images_train, y = labels_train, cv = 10)
accuracies.mean()
accuracies.std()

# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [
{'C': [1, 10, 100, 1000], 'gamma': [0.1, 0.3, 0.5, 0.7, 0.9]}]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 5,
                           n_jobs = -1)
grid_search = grid_search.fit(images_train, labels_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_


#Viewing an image from the dataset
i=10
img=test.iloc[i].as_matrix()
img=img.reshape((28,28))
plt.imshow(img,cmap='gray')
#plt.title(labels.iloc[i])

#Vieweing the histogram of the image
plt.hist(images_train.iloc[i])

#Converting the image from greyscale to pure black and white
images_test[images_test>0]=1
images_train[images_train>0]=1

#Optimization

def modelfit(alg, dtrain, predictors,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='auc', early_stopping_rounds=early_stopping_rounds, show_progress=False)
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain['Disbursed'],eval_metric='auc')
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
        
    #Print model report:
    print ("\nModel Report")
    print ("Accuracy : %.4g" % metrics.accuracy_score(dtrain['Disbursed'].values, dtrain_predictions))
    print ("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['Disbursed'], dtrain_predprob))
                    
    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    

#ANN

#Importing the keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

#Normalization
images_train = images_train / 255.0
labels_test = labels_test / 255.0

#Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
images_train = sc.fit_transform(images_train)
images_test = sc.transform(images_test)

# Reshape image in 3 dimensions (height = 28px, width = 28px , canal = 1)
images_train = images_train.values.reshape(-1,28,28,1)
labels_test = labels_test.values.reshape(-1,28,28,1)

#Initialising the ANN
classifier = Sequential()

#Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 393, init = 'uniform', activation = 'relu', input_dim = 784))

#Adding the 2nd hidden layer
classifier.add(Dense(output_dim = 393, init = 'uniform', activation = 'relu'))

#Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

#Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(images_train, labels_train, batch_size = 10, nb_epoch = 100)  

y_pred = classifier.predict(images_test)  
