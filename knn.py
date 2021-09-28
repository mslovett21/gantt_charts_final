import pandas as pd 
import numpy as np 
from sklearn.neighbors import KNeighborsClassifier
import argparse
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

def get_data(artifacts_dir):


    norm_train = pd.read_csv(artifacts_dir + 'train_normal_embeddings.csv')
    anom_train = pd.read_csv(artifacts_dir + 'train_anomaly_embeddings.csv')
 
    norm_test = pd.read_csv(artifacts_dir + 'test_normal_embeddings.csv')
    anom_test = pd.read_csv(artifacts_dir + 'test_anomaly_embeddings.csv')

    train_df = pd.concat([norm_train, anom_train])
    test_df  = pd.concat([norm_test, anom_test])

    return train_df, test_df

def knn_predictions(train_df, test_df,k):

    knn = KNeighborsClassifier(n_neighbors=k)
    train_feat, train_lab, test_feat, test_lab = train_df[train_df.columns[:-1]], train_df['target'], test_df[test_df.columns[:-1]], test_df['target']
    knn.fit(train_feat, train_lab)
    pred = knn.predict(test_feat)
    score = knn.score(test_feat, test_lab)

    return pred, test_lab, score

def plot_cm(lab, pred, artifacts_dir):

    cm = confusion_matrix(lab, pred)

    fig, ax = plt.subplots(figsize=(6,4))

    sns.heatmap(cm, annot=True, fmt='.2f', cmap = "YlGnBu")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')

    plt.title("KNN Confusion Matrix")
    plt.savefig(artifacts_dir + "knn_confusion_matrix.png")
    plt.close()

def main():

    parser = argparse.ArgumentParser('KNN')   
    parser.add_argument('-timestamp', help='Time string to locate folder containing embedding files.', default='20210927-175902')
    parser.add_argument('-k', type=int, help='k for KNN', default=3)    
    args = parser.parse_args()
    
    timestr = args.timestamp

    rel_path   = os.getcwd()
    dir_string = "AE_test"
    artifacts_dir = rel_path + '/artifacts/artifact_' + dir_string + "_" + timestr + '/'

    train_df, test_df = get_data(artifacts_dir)

    pred, true, score = knn_predictions(train_df, test_df,args.k)

    plot_cm(true, pred, artifacts_dir)
    prec, recall, fscore, _ = precision_recall_fscore_support(true, pred, average='binary')
    print("Prec: {}, recall: {}, fscore: {} \n".format(prec, recall, fscore))
    accuracy = accuracy_score(true,pred)
    print("Accuracy: {}".format(accuracy))

    return
    


if __name__ == "__main__":
    main()