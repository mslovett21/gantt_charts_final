import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import mlflow
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support



def plot_cm(lab, pred, epoch, additional_transformation, vis_results_path):
    target_names = ["cpu","hdd","loss","normal"]
    cm = confusion_matrix(lab, pred)
    # Normalise
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(6,4))
    sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=target_names, yticklabels=target_names, cmap = "YlGnBu")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    if additional_transformation :
        plt.title("VGG-16 with data augmentation")
    else:
        plt.title("VGG-16 without data augmentation")
    plt.savefig( vis_results_path + "/{}_confusion_matrix_norm.png".format(epoch))
    mlflow.log_artifact(vis_results_path + "/{}_confusion_matrix_norm.png".format(epoch))
    plt.close()


       
def draw_training_curves(train_losses, test_losses, curve_name,epoch, additional_transformation, vis_results_path):
    plt.clf()
    max_y = 2.0
    if curve_name == "Accuracy":
        max_y = 1.0
        plt.ylim([0,max_y])
        
    plt.xlim([0,epoch])
    plt.xticks(np.arange(0, epoch, 2))
    plt.plot(train_losses, label='Training {}'.format(curve_name))
    plt.plot(test_losses, label='Testing {}'.format(curve_name))
    plt.ylabel(curve_name)
    plt.xlabel('Epoch')
    if additional_transformation :
        plt.title("VGG-16 with data augmentation")
    else:
        plt.title("VGG-16 without data augmentation")
    plt.legend(frameon=False)
    plt.savefig(vis_results_path + "/{}_{}_final_vgg16.png".format(curve_name, epoch))
    mlflow.log_artifact(vis_results_path + "/{}_{}_final_vgg16.png".format(curve_name, epoch))
    plt.close()
