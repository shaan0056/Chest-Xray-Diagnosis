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
from helper import train, evaluate, calculate_AUROC
from plotter import plot_learning_curves, plot_confusion_matrix,plot_auc


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
BATCH_SIZE = 128
NUM_WORKERS = 4
NUM_EPOCHS = 2000
SAVE_FILE = "myCNN.pth"

# Path for saving model
PATH_OUTPUT = "../output/ChexPert/"
os.makedirs(PATH_OUTPUT, exist_ok=True)

transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])


dataset = ChexPertDataset(TRAIN_FILE,FILE_PATH,transform=transform)
valid_dataset = ChexPertDataset(VALID_FILE,FILE_PATH,transform=transform)

test_dataset,train_dataset = random_split(dataset,[2000,len(dataset) - 2000])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

#Model

model = DenseNet121(NUM_CLASSES)
model = torch.nn.DataParallel(model)

optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
criterion = torch.nn.BCELoss(size_average=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion.to(device)

best_val_loss = 0.0
train_losses, train_accuracies = [], []
valid_losses, valid_accuracies = [], []


for epoch in range(NUM_EPOCHS):
        train_loss, train_accuracy = train(model, device, train_loader, criterion, optimizer, epoch)
        valid_loss, valid_accuracy, valid_results = evaluate(model, device, valid_loader, criterion)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        if epoch == 0:
            best_val_loss = valid_loss

        is_best = valid_loss < best_val_loss

        if is_best:
            best_val_loss = valid_loss
            torch.save(model, os.path.join(PATH_OUTPUT, SAVE_FILE))

plot_learning_curves(train_losses, valid_losses)

best_model = torch.load(os.path.join(PATH_OUTPUT, SAVE_FILE))
test_loss, test_accuracy, test_results = evaluate(best_model, device, test_loader, criterion)
roc_score = calculate_AUROC(test_results[:, 0, :], test_results[:, 1, :], NUM_CLASSES)
plot_auc(test_results[:, 0, :], test_results[:, 1, :],NUM_CLASSES,CLASS_NAMES)





