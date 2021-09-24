from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse
import os


def plot_TSNE_2D(feature_df, artifacts_dir):
    
    tsne = TSNE(n_components=2, random_state=32, perplexity=80, metric='cosine')
    tsne_obj = tsne.fit_transform(feature_df)
    tsne_df = pd.DataFrame({'X': tsne_obj[:, 0],
                            'Y': tsne_obj[:, 1],
                            'labels': feature_df['target']})

    plt.figure(figsize=(10, 10))
    sns.scatterplot(x="X", y="Y",
                hue="labels",
                palette=['orange', 'blue'],
                legend='full',
                data=tsne_df)

    plt.savefig(artifacts_dir + 'tsne_embeddings.png')

    plt.close()

    return


def read_csv(artifacts_dir):

    normal_embedding_csv = artifacts_dir + 'normal_embeddings.csv'
    anomaly_embedding_csv = artifacts_dir + 'anomaly_embeddings.csv'

    normal_embed = pd.read_csv(normal_embedding_csv)
    anomaly_embed = pd.read_csv(anomaly_embedding_csv)

    df = normal_embed.append(anomaly_embed, ignore_index=True)

    return df


def main():

    parser = argparse.ArgumentParser('TSNE Visualization')   
    parser.add_argument('-timestamp', help='Time string to locate folder containing embedding files.', default='20210924-135625')
    args = parser.parse_args()
    
    timestr = args.timestamp

    rel_path   = os.getcwd()
    dir_string = "AE_test"
    artifacts_dir = rel_path + '/artifacts/artifact_' + dir_string + "_" + timestr + '/'

    embedding_df = read_csv(artifacts_dir)

    plot_TSNE_2D(embedding_df, artifacts_dir)

    return
    


if __name__ == "__main__":
    main()
