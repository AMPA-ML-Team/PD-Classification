import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.svm import SVC
from operator import itemgetter

df = pd.read_csv('./pd_speech_features.csv')

label = pd.Series(df.iloc[1:, -1].values, name=df.iloc[0, -1]).astype(np.int8)
patientId = pd.Series(df.iloc[1:, 0].values,
    name=df.iloc[0, 0]).astype(np.int16)
gender = pd.Series(df.iloc[1:, 1].values,
    name=df.iloc[0, 1]).astype(np.int8)
baselineFeats = pd.DataFrame(
    df.iloc[1:, 2:23].values, columns=df.iloc[0, 2:23]).astype(np.float64)
intensityFeats = pd.DataFrame(
    df.iloc[1:, 23:26].values, columns=df.iloc[0, 23:26]).astype(np.float64)
formatFeats = pd.DataFrame(
    df.iloc[1:, 26:30].values, columns=df.iloc[0, 26:30]).astype(np.float64)
bandwidthFeats = pd.DataFrame(
    df.iloc[1:, 30:34].values, columns=df.iloc[0, 30:34]).astype(np.float64)
vocalFeats = pd.DataFrame(
    df.iloc[1:, 34:56].values, columns=df.iloc[0, 34:56]).astype(np.float64)
mfccFeats = pd.DataFrame(
    df.iloc[1:, 56:140].values, columns=df.iloc[0, 56:140]).astype(np.float64)
waveletFeats = pd.DataFrame(
    df.iloc[1:, 140:322].values, columns=df.iloc[0, 140:322]).astype(np.float64)
tqwtFeats = pd.DataFrame(
    df.iloc[1:, 322:-1].values, columns=df.iloc[0, 322:-1]).astype(np.float64)

data = {"patientId": patientId,
            "gender": gender,
            "baselineFeats": baselineFeats,
            "intensityFeats": intensityFeats,
            "formantFeats": formatFeats,
            "bandwidthFeats": bandwidthFeats,
            "vocalFeats": vocalFeats,
            "mfccFeats": mfccFeats,
            "waveletFeats": waveletFeats,
            "tqwtFeats": tqwtFeats,
            "label": label}

features = ["gender", "baselineFeats", "intensityFeats", "formantFeats",
                "bandwidthFeats", "vocalFeats", "mfccFeats", "waveletFeats", "tqwtFeats"]


def convert_data(data, features):
    if len(features) == 1:
        return data[features[0]]
    return pd.concat(itemgetter(*features)(data), axis=1).values




def SVM_PD(params):
    X = convert_data(data, features)
    y = data['label']
    folds = KFold(n_splits=params['number_of_folds'], shuffle=True)
    scores = {'accuracy': [], 'fscore': [], 'precision': [], 'recall': []}
    conf_mats = []
    for train_idx, test_idx in folds.split(X):
        x_train, x_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = MinMaxScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)


        svm = SVC(kernel=params['kernel'], degree=params['degree_of_svm'],
                  C=params['c'], shrinking=params['shrinking'])
        svm.fit(x_train, y_train)
        preds = svm.predict(x_test)
        y_pred = np.round(preds)
        scores['accuracy'].append(accuracy_score(y_test, y_pred))
        scores['fscore'].append(f1_score(y_test, y_pred))
        conf_mats.append(confusion_matrix(y_test, y_pred))
    return [scores, conf_mats]


def running_SVM(params):
    best_result = 0
    best_fscore = 0
    sum = 0
    number_of_runs = params['number_of_runs']
    for i in range(number_of_runs):
        scores, confs = SVM_PD(params)
        x = 100 * np.mean(scores['accuracy'])
        fscore = 100 * np.mean(scores['fscore'])
        sum = sum + x
        if x > best_result:
            best_result = x
            best_fscore = fscore
            best_confs = confs
        print(f"{i + 1}th run completed...")

    average = sum/number_of_runs

    print("******************")
    print(f"ACCURACY: {best_result:1.2f}")
    print(f"F_SCORE: {best_fscore:1.2f}" )
    print(f"Average accuracy on the {params['number_of_runs']} runs: {average: 1.2f}")
    print("******************")
    print("CONFUSION MATRICES:")
    for i in range(len(best_confs)):
        print(best_confs[i])
    print("******************")

    print(f"Degree of SVM: {params['degree_of_svm']}")
    print(f"Number of Folds: {params['number_of_folds']}")
    print(f"Number of runs: {params['number_of_runs']}")


params = {'kernel' : 'poly',
          'degree_of_svm' : 23,
          'shrinking' : True,
          'c' : 1,
          'number_of_folds' : 5,
          'number_of_runs' : 2500}


if __name__ == "__main__":
    running_SVM(params)