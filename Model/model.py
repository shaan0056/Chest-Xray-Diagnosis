import os
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader
from ChexPertDataset import ChexPertDataset
from DenseNet121 import DenseNet121
import torch.optim as optim
from helper import train, evaluate, calculate_auc, predict_pathology, save_training_data
from plotter import plot_learning_curves, plot_confusion_matrix,plot_auc
import argparse


parser = argparse.ArgumentParser(description='Uncertainity')
parser.add_argument('--U', type=str,
                    help='An optional integer argument')
parser.add_argument('--competition', action='store_true',
                    help='A boolean switch')
parser.add_argument('--freeze', action='store_true',
                    help='A boolean switch')

args = parser.parse_args()



torch.manual_seed(42)
if torch.cuda.is_available():
        torch.cuda.manual_seed(42)


FILE_PATH = "../Data/CheXpert-v1.0-small/"
TRAIN_FILE = "train.csv"
VALID_FILE = "valid.csv"
NUM_CLASSES = 14
CLASS_NAMES = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
               'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax',
               'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']
BATCH_SIZE = 16
NUM_WORKERS = 16
NUM_EPOCHS = 3
SAVE_FILE = "myCNN.pth"

# Path for saving model
PATH_OUTPUT = "../output/ChexPert/"
os.makedirs(PATH_OUTPUT, exist_ok=True)

COMPETITION = False
if args.competition:
    COMPETITION = True

transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])

if args.U == "ones":
    dataset = ChexPertDataset(TRAIN_FILE,FILE_PATH,transform=transform,uncertainity="ones")
else:
    dataset = ChexPertDataset(TRAIN_FILE, FILE_PATH, transform=transform)

valid_dataset = ChexPertDataset(VALID_FILE,FILE_PATH,transform=transform)

test_dataset,train_dataset = random_split(dataset,[500,len(dataset) - 500])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

#Model

if args.freeze:
    model = DenseNet121(NUM_CLASSES,True)
else:
    model = DenseNet121(NUM_CLASSES,False)


model = torch.nn.DataParallel(model)

optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
criterion = torch.nn.BCELoss(size_average=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion.to(device)

best_val_auc = 0.6
train_losses, train_accuracies = [], []
valid_losses, valid_accuracies = [], []


for epoch in range(NUM_EPOCHS):
        train_loss, train_auc = train(model, device, train_loader, criterion, optimizer, epoch, NUM_CLASSES, COMPETITION,PATH_OUTPUT)
        valid_loss, valid_auc, valid_results = evaluate(model, device, valid_loader, criterion, NUM_CLASSES, COMPETITION)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        train_accuracies.append(train_auc)
        valid_accuracies.append(valid_auc)

        is_best = valid_auc > best_val_auc
        if is_best:
            best_val_acc = valid_auc
            torch.save(model, os.path.join(PATH_OUTPUT, SAVE_FILE))



save_training_data(train_losses, valid_losses, train_accuracies, valid_accuracies)
plot_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies)

best_model = torch.load(os.path.join(PATH_OUTPUT, SAVE_FILE))
test_loss, test_accuracy, test_results = evaluate(best_model, device, test_loader, criterion,NUM_CLASSES,COMPETITION)
roc_score = calculate_auc(test_results[:, 0, :], test_results[:, 1, :], NUM_CLASSES)
plot_auc(test_results[:, 0, :], test_results[:, 1, :],NUM_CLASSES,CLASS_NAMES,args.U)

#print(predict_pathology(model,device,test_loader))





