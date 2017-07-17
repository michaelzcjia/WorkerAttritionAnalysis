import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk
from sklearn import ensemble
from sklearn import linear_model
from sklearn import tree
from sklearn import svm
from sklearn.externals import joblib

sys.path.append("../DataProcessing")
sys.path.append("../Data")
sys.path.append('../Models')
sys.path.insert(0, '../DataProcessing/DataRetriever.py')
import DataCleansing

A = ensemble.RandomForestClassifier();
B = ensemble.AdaBoostClassifier()
C = svm.SVC()
Models={} #Dictionary of regressor models
Models["RForest"] = A
Models["Bayesian Ridge"] = B
Models["SVC"] = C

def train_test_models(trainFeat, trainLab, testFeat, testLab):
    global FigureNumber

    Best_Score = 0;
    Best_Model = A;
    Best_Model_Name = "RForest"
    Scores = []


    xplot = [x for x in range(len(testLab))]

    for modelName, model in Models.items():
        model.fit(trainFeat, trainLab) #Trains the model
        score = model.score(testFeat, testLab) #gets a score for a test
        #Determines the best socring model
        Scores.append((modelName,[score]))
        if (score > Best_Score):
            Best_Score = score
            Best_Model = model
            Best_Model_Name = modelName

    prediction = Best_Model.predict(testFeat) #Predicts based on features
    with open('../Models/ScorePredictionModel.pkl', 'wb') as f:
        joblib.dump(Best_Model, f)

    df = pd.DataFrame.from_items(items=Scores,orient='index',columns=['Score'])
    df.plot(kind ='bar' ,ylim =(0.2,1.0), figsize=(13,6),align='center',colormap='Accent')
    plt.xticks(np.arange(len(Models)),df.index)
    plt.title( "Compare Model Predictions")
    FigureNumber +=1

    plt.figure(FigureNumber,figsize=(15,6))
    plt.plot(xplot,testLab,'r*',label = 'Actual Score')
    plt.legend(bbox_to_anchor=(1, 1), loc=2);

    plt.plot(xplot,prediction,'g*', label = 'Prediction Score')
    plt.legend(bbox_to_anchor=(1, 1), loc=2);

    plt.xlabel("Game")
    plt.ylabel("Score")

    plt.title( Best_Model_Name + " Score Prediction: " + str(Best_Score))

    return Best_Model

def plotFeatureImportance(model, featuresDF):
    global FigureNumber

    features = list(featuresDF.columns)
    featImportance = list(model.feature_importances_)
    print(featImportance)
    print(features)
    xplot = [x for x in range(len(features))]
    print(xplot)
    FigureNumber +=1
    plt.figure(FigureNumber)
    plt.xticks()
    plt.plot(kind = "bar", x = xplot, y = featImportance)


if __name__ == '__main__':
    global FigureNumber
    FigureNumber = 1
    dr = DataCleansing.DataFrame()
    normData = dr.normalize()
    training, testing = dr.getFeaturesLabels(normData)
    trainFeat = training[0]
    trainLab = training[1]
    testFeat = testing[0]
    testLab = testing[1]
    bestModel = train_test_models(trainFeat, trainLab, testFeat, testLab)
    plotFeatureImportance(bestModel, trainFeat)
    plt.show()
