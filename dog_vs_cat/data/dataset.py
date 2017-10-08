# coding= utf-8
import os
from PIL import Image
from torch.utils import data
import numpy as np
from torchvision import transforms as T

class DogVsCatDataset(data.Dataset):
    """
    Describe:
        load dog vs cat image to dateset
    """
    def __init__(self, data_root = "/home/xksj/Data/lp/dog_vs_cat/", transform = None, is_train = True, is_test = False):
        """
        Describe:
            None
        Args:
            data_root: the path you put data in
            transform: the data transform
            is_train: True is the data is loaded for train
            is_test:
        """
        self.test = is_test


        """
        test and train data has different path formate
        test: data_root/test/1000.jpg
        train: data_root/train/cat.1000.jpg
        """
        if self.test:
            data_root = os.path.join(data_root, "test")
            imgs = [os.path.join(data_root, img) for img in os.listdir(data_root)]
            self.imgs = sorted(imgs, key=lambda x: int(x.split("/")[-1].split(".")[-2]))
        else:
            data_root = os.path.join(data_root, "train")
            imgs = [os.path.join(data_root, img) for img in os.listdir(data_root)]
            self.imgs = sorted(imgs, key=lambda x: int(x.split(".")[-2]))
        # judge if train or val
        # split train : val = 7 : 3
        data_len = len(self.imgs)
        if is_train:
            self.imgs = self.imgs[:int(0.7 * data_len)]
        else:
            self.imgs = self.imgs[int(0.7 * data_len):]
        self.data_len = len(self.imgs)
        # transform and data augment
        if transform is None:
            normalize = T.Normalize(mean = [0.485, 0.456, 0.406],
                                    std = [0.229, 0.224, 0.225])
            # train data need data augment

            # 测试集和验证集不用数据增强
            if self.test or not is_train:
               self.transforms = T.Compose([
                   T.Scale(224),
                   T.CenterCrop(224),
                   T.ToTensor(),
                   normalize
               ])
           # 训练集需要数据增强
            else:
                self.transforms = T.Compose([
                    T.Scale(256),
                    T.RandomSizedCrop(224),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    normalize
                ])
    def __getitem__(self, index):
        """

        """
        img_path = self.imgs[index]
        if self.test:
            label = int(img_path.split("/")[-1].split(".")[-2])
        else:
            label = 1 if "dog" in img_path.split("/")[-1] else 0

        data = self.transforms(Image.open(img_path))
        return data, label

    def __len__(self):
        return len(self.imgs)
if __name__ == "__main__":
    data_set = DogVsCatDataset(data_root = "/home/xksj/Data/lp/dog_vs_cat/", transform = None, is_train = True, is_test = False)
    print(data_set[1][1])
