
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


def plot_learning_curves(train_losses, valid_losses,train_auc,valid_auc):



      plt.plot(np.arange(len(train_losses)),train_losses)
      plt.plot(np.arange(len(valid_losses)), valid_losses)
      plt.title("Loss Curve")
      plt.xlabel("epoch")
      plt.ylabel("Loss")
      plt.tight_layout()
      plt.legend(["Training Loss","Validation Loss"],loc='upper right')
      plt.savefig("Loss-Curve.jpg")
      plt.clf()

      plt.plot(np.arange(len(train_auc)), train_auc)
      plt.plot(np.arange(len(valid_auc)), valid_auc)
      plt.title("AUC Curve")
      plt.xlabel("epoch")
      plt.ylabel("AUC")
      plt.legend(["Training AUC", "Validation AUC"], loc='upper right')
      plt.tight_layout()
      plt.savefig("AUC-Curve.jpg")
      plt.clf()

      pass


def plot_confusion_matrix(results, class_names):

      y_true = [x[0] for x in results]
      y_pred = [x[1] for x in results]
      plot_confusion_matrixx(y_true, y_pred, classes=class_names, normalize=True,
                            title='Normalized confusion matrix')
      plt.savefig("Confusion-Matrix.jpg")
      pass

def plot_confusion_matrixx(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    #reference - https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')


    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")


    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def plot_auc(ground_truth,prediction,num_classes,class_names,Uncertainity):

    colors = ['black', 'brown', 'red', 'green', 'blue', 'turquoise', 'purple', 'orange', 'navy', 'lime', 'pink',
              'violet','olive','tan']

    for i in range(num_classes):

        fpr, tpr, threshold = metrics.roc_curve(ground_truth[:, i], prediction[:, i])
        roc_auc = metrics.auc(fpr, tpr)

        plt.title('ROC Curve ')
        plt.plot(fpr, tpr, label='U-{}: AUC = %0.2f'.format("ones" if Uncertainity == "ones" else "zeros") % roc_auc,color=colors[i])

        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')

    lines = [Line2D([0], [0], color=c, linewidth=3) for c in colors]
    plt.legend(lines,class_names,loc='lower right')
    plt.savefig("ROC_Curve.png", dpi=1000)
    plt.clf()

def plot_images(data_loader):

    real_batch = next(iter(data_loader))
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(
        np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))
    plt.savefig("image_grid.png", dpi=1000)


def plot_gan_loss(G_losses,D_losses):

    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("loss_gan.png", dpi=1000)