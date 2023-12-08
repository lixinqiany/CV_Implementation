import torch
from torch.utils.data import Dataset, DataLoader
import os, tqdm,time
import numpy as np
import matplotlib.pyplot as plt


class MNIST_Dataset(Dataset):
    def __init__(self):
        self.files = self.checkDataFile()
        self.dataPath = os.path.join(os.getcwd(),'data')
        self.data = self.prepareData(self.files)


    def __getitem__(self, item):
        pass


    def __len__(self):
        pass


    def checkDataFile(self):
        workSpace = os.getcwd()
        fileSpace = os.path.join(workSpace,'data')
        files = os.listdir(fileSpace)
        files = [x for x in files if 'zip' not in x]
        print("Current working directory -> "+workSpace)
        print("The dataset is stored in -> "+ fileSpace+"\nlisted as follows:")
        for x in files:
            print("    "+x)

        return files


    def extractData(self, file, key):
        byteData = np.fromfile(os.path.join(self.dataPath, file), dtype=np.uint8)
        if "label" in key:
            # 0004 four digits -> how many images?
            imageNum = byteData[4]*2**24+byteData[5]*2**16+byteData[6]*2**8+byteData[7]
            offset = 8
            labels = byteData[offset:]
            return labels
        else:
            imageNum = byteData[4]*2**24+byteData[5]*2**16+byteData[6]*2**8+byteData[7]
            width = byteData[12]*2**24+byteData[13]*2**16+byteData[14]*2**8+byteData[15]
            height = byteData[8]*2**24+byteData[9]*2**16+byteData[10]*2**8+byteData[11]
            offset = 16
            imgaes = byteData[offset:]
            images = imgaes.reshape((imageNum, width, height, 1))

            return images

    def prepareData(self, files):
        data = {"tarin":None, "train_label":None, "test":None, "test_label":None}
        text = None
        progress = tqdm.tqdm(files)
        for file in progress:
            if "train" in file:
                if "label" in file:
                    text = "train_label"
                else:
                    text = "train"
            else:
                if "label" in file:
                    text = "test_label"
                else:
                    text = "test"
            data[text] = self.extractData(file, text)
            progress.set_description("extract -> "+file)

        return data


test = MNIST_Dataset()
