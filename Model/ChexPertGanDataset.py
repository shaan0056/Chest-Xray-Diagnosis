
import torch
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

class ChexPertGanDataset(Dataset):

    def __init__(self,image_csv,file_path,select_class,transform=None,uncertainity=None):

        """
        Args:
        image_csv (string): train or test csv containing Chest Xray image path name and corresponding labels
        file_path (string): path containing image_csv
        select_class(list) : selected classes for GAN training
        transform : Any image transformation to be applied to the image
        """

        image_csv_df = pd.read_csv(file_path+image_csv)

        for classes in select_class:

            image_csv_df = image_csv_df[image_csv_df[classes] == 1]


        image_name = image_csv_df.ix[:,0].tolist()
        self.image = ["../Data/" + image for image in image_name]
        image_labels = image_csv_df.ix[:,5:].fillna(0)

        for col in image_labels:
            if uncertainity is None:
                image_labels[col].replace([-1],0,inplace=True)
            if uncertainity == "ones":
                image_labels[col].replace([-1], 1, inplace=True)

        self.labels = image_labels.values
        self.transform = transform


    def __getitem__(self, item):

        image = self.image[item]
        image = Image.open(image).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        label = self.labels[item]
        return image,torch.FloatTensor(label)


    def __len__(self):

        return len(self.image)







