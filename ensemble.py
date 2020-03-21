"""
    This is the LogisticRegression stacking ensemble code.
    Put the outputs of your desired models to ensemble in a folder named 'models'
    in the current directory.
"""


import pandas as pd
import numpy as np
import scipy.stats as ss

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

import os
import sys


def load_data(file_name):
    df = pd.read_csv(file_name)

    y = pd.Series(df.iloc[1:, -1].values, name=df.iloc[0, -1]).astype(np.int8)
    patient_id = pd.Series(df.iloc[1:, 0].values,
                           name=df.iloc[0, 0]).astype(np.int16)
    gender = pd.Series(df.iloc[1:, 1].values,
                       name=df.iloc[0, 1]).astype(np.int8)

    baseline_feats = pd.DataFrame(
        df.iloc[1:, 2:23].values, columns=df.iloc[0, 2:23]).astype(np.float64)
    intensity_feats = pd.DataFrame(
        df.iloc[1:, 23:26].values, columns=df.iloc[0, 23:26]).astype(np.float64)
    format_feats = pd.DataFrame(
        df.iloc[1:, 26:30].values, columns=df.iloc[0, 26:30]).astype(np.float64)
    bandwidth_feats = pd.DataFrame(
        df.iloc[1:, 30:34].values, columns=df.iloc[0, 30:34]).astype(np.float64)
    vocal_feats = pd.DataFrame(
        df.iloc[1:, 34:56].values, columns=df.iloc[0, 34:56]).astype(np.float64)
    mfcc_feats = pd.DataFrame(
        df.iloc[1:, 56:140].values, columns=df.iloc[0, 56:140]).astype(np.float64)
    wavelet_feats = pd.DataFrame(
        df.iloc[1:, 140:322].values, columns=df.iloc[0, 140:322]).astype(np.float64)
    tqwt_feats = pd.DataFrame(
        df.iloc[1:, 322:-1].values, columns=df.iloc[0, 322:-1]).astype(np.float64)

    return {"patientId": patient_id,
            "gender": gender,
            "baselineFeats": baseline_feats,
            "intensityFeats": intensity_feats,
            "formantFeats": format_feats,
            "bandwidthFeats": bandwidth_feats,
            "vocalFeats": vocal_feats,
            "mfccFeats": mfcc_feats,
            "waveletFeats": wavelet_feats,
            "tqwtFeats": tqwt_feats,
            "label": y}


def convert_data(data, features):
    if len(features) == 1:
        return data[features[0]]
    return pd.concat(itemgetter(*features)(data), axis=1)


def __main__(dataset_filename):
    data = load_data(dataset_filename)
    y = pd.concat([data["patientId"], data['label']], axis=1)
    # y = data['label']

    models = {}
    for filename in os.listdir("./models/"):
        df = pd.read_csv(f'./models/{filename}')
        models[filename] = df['y'].values

    folds = KFold(n_splits=5, shuffle=True)
    pred = []
    X = pd.DataFrame(data=models, columns=models.keys())

    best_score = (0, 0)
    avg_score = [0, 0]
    best_pred = []
    for i in range(50):
        pred = []
        for train_idx, test_idx in folds.split(X):
            x_train, x_test = X.iloc[train_idx, :], X.iloc[test_idx, :]
            y_train, y_test = y.iloc[train_idx, 1], y.iloc[test_idx, 1]
            ids = y.iloc[test_idx, 0]

            lr = LogisticRegression(penalty='l1', solver='liblinear')
            lr.fit(x_train, y_train)

            preds = lr.predict(x_test)
            y_pred = np.round(preds)
            y_pred = pd.DataFrame(data={'id': ids, 'y': y_pred})
            print(y_pred['y'].unique())
            pred.append(y_pred)

        pred = pd.concat(pred, axis=0).sort_values(by=['id'])['y']
        y2 = y['class']

        ''' Uncomment in case you want to enable label voting. '''
        # pred = pd.DataFrame(data={'id': data['patientId'], 'y': pred}, columns=['id', 'y']).groupby(
        #     ['id']).agg(lambda x: x.value_counts().index[0])['y']
        # y2 = y.groupby(['id']).agg(
        #     lambda x: x.value_counts().index[0])

        print(len(y2), len(pred))
        print(classification_report(y2, pred))
        print("ACCURACY =", accuracy_score(y2, pred))
        print("FSCORE =", f1_score(y2, pred))
        print("PRECISION =", precision_score(y2, pred))
        print("RECALL =", recall_score(y2, pred))
        # print(confusion_matrix(y, pred))
        acc = accuracy_score(y2, pred)
        fscore = f1_score(y2, pred)
        avg_score[0] += acc
        avg_score[1] += fscore
        if acc > best_score[0]:
            best_score = (acc, fscore)
            best_pred = pred

    print('Best accuracy and F1 score in 50 runs:', best_score)
    print('Average accuracy and F1 score in 50 runs:',
          avg_score[0] / 50, avg_score[1] / 50)


if __name__ == "__main__":
    __main__(sys.argv[1])
