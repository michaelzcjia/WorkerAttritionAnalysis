import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append("../Data")

class DataFrame():
    def __init__ (self, file_loc = '../data/HR_comma_sep.csv'):
        '''initialize dataframe from csv'''

        df = pd.read_csv(file_loc)
        self.df = df
        self.label = df["left"]

    def normalize(self):
        df = self.df
        #Classify salary 1,2,3 = low,med,high
        salaryClas = list(df["salary"].unique())
        for i in range(1,4):
            df.replace(salaryClas[i-1], value = i, inplace = True)

        #One hot encoding for employee position
        df = pd.get_dummies(df, columns=['sales'])

        #Normalize all columns
        for feature_name in df.columns:
            max = df[feature_name].max()
            min = df[feature_name].min()
            df[feature_name] = (df[feature_name] - min) / (max - min)

        return df

    def getFeaturesLabels(self, df, pct=0.7, dropColumns = None):
        if dropColumns != None:
            df = df.drop(dropColumns)

        pctNum = int(np.floor(pct * len(df)))

        label = df["left"]
        trainLb = label.iloc[:pctNum]
        testLb = label.iloc[pctNum:]

        #print(df)
        features = df.drop(labels = "left", axis = 1)
        trainFt = features.iloc[:pctNum]
        testFt = features.iloc[pctNum:]

        return ((trainFt, trainLb), (testFt, testLb))


if __name__ == "__main__":
    data = DataFrame()
    x = data.normalize()
   # print(x)
    y = data.getFeaturesLabels(x, 0.7)
    print(y)
