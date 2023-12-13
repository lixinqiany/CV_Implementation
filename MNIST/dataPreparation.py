from torch.utils.data import Dataset
import os, tqdm
import numpy as np
from torchvision import transforms


class MNIST_Dataset(Dataset):
    def __init__(self, transform=False, trained=True):
        self.files = self.checkDataFile()
        self.dataPath = os.path.join(os.getcwd(), 'data')
        self.data = self.prepareData(self.files)
        self.trained = trained
        self.num = None
        self.mean = 0
        self.std = 0
        self.meanAndStd()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ]) if transform else None

    def __getitem__(self, index: int):
        if self.trained:
            samples = self.data["train"]
            labels = self.data["train_label"]
            sample = self.transform(samples[index]) if self.transform is not None else samples[index]
            labels = labels[index]
            return sample, labels
        else:
            samples = self.data["test"]
            labels = self.data["test_label"]
            sample = self.transform(samples[index]) if self.transform is not None else samples[index]
            labels = labels[index]
            return sample, labels

    def __len__(self):
        if self.trained:
            return self.data["train"].shape[0]
        else:
            return self.data["test"].shape[0]

    def checkDataFile(self):
        workSpace = os.getcwd()
        fileSpace = os.path.join(workSpace, 'data')
        files = os.listdir(fileSpace)
        files = [x for x in files if 'zip' not in x]
        print("Current working directory -> " + workSpace)
        print("The dataset is stored in -> " + fileSpace + "\nlisted as follows:")
        for x in files:
            print("    " + x)

        return files

    def extractData(self, file, key):
        byteData = np.fromfile(os.path.join(self.dataPath, file), dtype=np.uint8)
        if "label" in key:
            # 0004 four digits -> how many images?
            imageNum = byteData[4] * 2 ** 24 + byteData[5] * 2 ** 16 + byteData[6] * 2 ** 8 + byteData[7]
            self.num = imageNum
            offset = 8
            labels = byteData[offset:]
            return labels
        else:
            imageNum = byteData[4] * 2 ** 24 + byteData[5] * 2 ** 16 + byteData[6] * 2 ** 8 + byteData[7]
            self.num = imageNum
            width = byteData[12] * 2 ** 24 + byteData[13] * 2 ** 16 + byteData[14] * 2 ** 8 + byteData[15]
            height = byteData[8] * 2 ** 24 + byteData[9] * 2 ** 16 + byteData[10] * 2 ** 8 + byteData[11]
            offset = 16
            images = byteData[offset:]
            images = images.reshape((imageNum, width, height, 1))

            return images

    def prepareData(self, files):
        data = {"tarin": None, "train_label": None, "test": None, "test_label": None}
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
            progress.set_description("extract -> " + file)

        return data

    def meanAndStd(self):
        data = self.data['train']
        self.mean = 0
        self.std = 0
        for i in range(data.shape[0]):
            img = transforms.ToTensor()(data[i])
            img = img.view(1, -1)
            self.mean += img.mean(1)
            self.std += img.std(1)
        self.mean = self.mean / data.shape[0]
        self.std = self.std / data.shape[0]
