
import torch
import pandas as pd
from torch.utils.data import Dataset
import cv2
from PIL import Image

class ChexPertDataset(Dataset):

    def __init__(self,image_csv,file_path,transform=None):

        """
        Args:
        image_csv (string): train or test csv containing Chest Xray image path name and corresponding labels
        file_path (string): path containing image_csv
        transform : Any image transformation to be applied to the image
        """

        image_csv_df = pd.read_csv(file_path+image_csv)
        image_name = image_csv_df.ix[:,0].tolist()
        self.image = ["../Data/" + image for image in image_name]
        image_labels = image_csv_df.ix[:,5:].fillna(0)

        for col in image_labels:
            image_labels[col].replace([-1],0,inplace=True)

        self.labels = image_labels.values
        self.transform = transform


    def __getitem__(self, item):

        image = self.image[item]
        #grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = Image.open(image).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        label = self.labels[item]
        return image,torch.FloatTensor(label)


    def __len__(self):

        return len(self.image)







