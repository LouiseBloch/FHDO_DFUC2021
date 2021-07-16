import pandas as pd
import torch
import numpy as np
import cv2
import os

def prepare_Valid_Dataset(data_dir):
    validDataset=os.listdir(data_dir)
    validDataset=pd.DataFrame(validDataset,columns=["image"])
    validDataset["image_path"]=data_dir+validDataset["image"]
    return validDataset
    
def initialize_DFUC_Inference_Dataloader():
    class DFUCDataset(torch.utils.data.Dataset):
        def __init__(self, data, transform=None):
            self.data=data
            self.transform = transform

        def __len__(self):
            return self.data.shape[0]
        #load and transform image and load label of one given sample by given index
        def __getitem__(self, index):
            #load image 
            img_obj = self.data.iloc[index]
            img = cv2.imread(img_obj.image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            #img = Image.open(img_obj.image_path).convert("RGB")
            #load label
            #y_label = torch.tensor(img_obj.class_id)
            #apply transformation to image
            if self.transform is not None:
                img = self.transform(image=img)["image"]

            return (img)
    return(DFUCDataset)

def prepare_Training_Dataset(TRAIN_LABELS_DIR,DATA_DIR):
    #read csv of containining training data set
    trainingDataset=pd.read_csv(TRAIN_LABELS_DIR)
    #remove all images with no label
    trainingDataset=trainingDataset[trainingDataset.image.str.startswith('30')]
    #set image paths
    trainingDataset["image_path"]=DATA_DIR+trainingDataset.image
    #convert class predictions to number between 0 and 3
    trainingDataset["class_id"]=np.argmax(trainingDataset.filter(["none","infection","ischaemia","both"]).to_numpy(),axis=1)
    return(trainingDataset)

def get_number_of_classes_in_dataset(dataset):
    return(dataset["class_id"].value_counts().shape[0])

def initialize_DFUC_Dataloader():
    #Dataset loader
    class DFUCDataset(torch.utils.data.Dataset):
        def __init__(self, data, transform=None):
            self.data=data
            self.transform = transform
        def __len__(self):
            return self.data.shape[0]
        #load and transform image and load label of one given sample by given index
        def __getitem__(self, index):
            #load image 
            img_obj = self.data.iloc[index]
            img = cv2.imread(img_obj.image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            #img = Image.open(img_obj.image_path).convert("RGB")
            #load label
            y_label = torch.tensor(img_obj.class_id)
            #apply transformation to image
            if self.transform is not None:
                img = self.transform(image=img)["image"]

            return (img, y_label)
    return(DFUCDataset)

def load_dataset(datasetDf,DFUCDataset,transform,shuffle,batch_size,use_oversampling):
    dataset = DFUCDataset(datasetDf,transform=transform)
    if use_oversampling==False:
        loader = torch.utils.data.DataLoader(dataset=dataset, shuffle=shuffle, batch_size=batch_size)
    if use_oversampling==True:
        targets = datasetDf.class_id
        class_count = np.unique(targets, return_counts=True)[1]

        weight = 1. / class_count
        samples_weight = weight[targets]
        samples_weight = torch.from_numpy(samples_weight)

        sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight, len(samples_weight))
        loader = torch.utils.data.DataLoader(dataset=dataset, sampler=sampler, batch_size=batch_size)
    return(loader)