import pandas as pd 
import numpy as np 
from sklearn.neighbors import KNeighborsClassifier
import argparse
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import seaborn as sns
import matplotlib.pyplot as plt

def get_data(artifacts_dir):

    normal_embedding_csv = artifacts_dir + 'normal_embeddings.csv'
    anomaly_embedding_csv = artifacts_dir + 'anomaly_embeddings.csv'

    normal_embed = pd.read_csv(normal_embedding_csv)
    anomaly_embed = pd.read_csv(anomaly_embedding_csv)
 
    norm_train, norm_test = train_test_split(normal_embed, test_size=0.2)
    anom_train, anom_test = train_test_split(anomaly_embed, test_size=0.2)

    train_df = pd.concat([norm_train, anom_train])
    test_df = pd.concat([norm_test, anom_test])

    return train_df, test_df

def get_all_gant_data():

    path = '/home/shubham/crisis-computing/SupContrast/supcon_gant_embeddings.csv'
    embed = pd.read_csv(path)
    train_embed, test_embed = train_test_split(embed, test_size=0.3)

    return train_embed, test_embed

    
def knn_predictions(train_df, test_df):

    knn = KNeighborsClassifier(n_neighbors=2)
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
    args = parser.parse_args()
    
    timestr = args.timestamp

    rel_path   = os.getcwd()
    dir_string = "AE_test"
    artifacts_dir = rel_path + '/artifacts/artifact_' + dir_string + "_" + timestr + '/'

    train_df, test_df = get_data(artifacts_dir)

    supcon_train, supcon_test = get_all_gant_data()
    pred, true, score = knn_predictions(supcon_train, supcon_test)
    supcon_path = '/home/shubham/crisis-computing/SupContrast/'
    plot_cm(true, pred, supcon_path)

    prec, recall, fscore, _ = precision_recall_fscore_support(true, pred, average='macro')
    print("Prec: {}, recall: {}, fscore: {} \n".format(prec, recall, fscore))
    
    return
    


if __name__ == "__main__":
    main()