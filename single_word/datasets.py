from torch.utils.data import Dataset
from torchvision import transforms
from IPython.display import display
from torchvision.transforms import ToPILImage
toPIL = ToPILImage()
import torch
torch.cuda.set_device(0)
import cv2, numpy as np
from PIL import Image
from utils import Tools
tools = Tools('data/')

class Single_word(Dataset):
    def __init__(self, length, real_dir=None):
        self.len = length
        self.transfrom = transforms.Compose([
            transforms.Resize(128),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5]) 
        ])
        self.real_dir = real_dir
 
    def __len__(self):
        return self.len
 
    def __getitem__(self, idx):
        if not self.real_dir:
            image, label = tools.get_sample()
        else:
            image, label = tools.get_real_sample(real_dir=self.real_dir)
        image = self.transfrom(image)
        label = torch.from_numpy(np.array(label))
        return image, label
    def evaluate(self, model=None, total=1000):
        cnt = 0
        total = int(abs(total))
        i = total
        while i:
            i -= 1
            image, label = self.__getitem__(0)
            predict = model(image.reshape(1,1,tools.cfg.BG_SIZE,tools.cfg.BG_SIZE))
            index = predict.argmax()
            if index == label: cnt+=1
        print(f'Tested with {total} images.\nPrecision is {100*cnt/total :.2f}%')

    def test_img(self, model=None, img_path=None):
        if img_path:
            img = Image.open(img_path)
            img = np.array(img)
            img = tools.get_binary_img(img)
            img = Image.fromarray(img)
            img = self.transfrom(img)
        else:
            img, _ = self.__getitem__(0)
        predict = model(img.reshape(1,1,tools.cfg.BG_SIZE,tools.cfg.BG_SIZE))
        index = predict.argmax()
        print('Predicted as ' + tools.cfg.ch_list[index])
        image = img.clone()  # clone the tensor
        image = image.squeeze(0)  # remove the fake batch dimension
        image = toPIL(image)
        display(image)
        return image

class Real_word(Dataset):
    def __init__(self, path):
        self.path = path
        self.transfrom = transforms.Compose([
            transforms.Resize(128),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5]) 
        ])
        self.img_names = os.listdir(path)
        self.len = len(self.img_names)
 
    def __len__(self):
        return self.len
 
    def __getitem__(self, idx):
        img = Image.open(self.path+self.img_names[idx])
        img = np.array(img)
        img = tools.get_binary_img(img)
        img = tools.make_noise(img)
        img = Image.fromarray(img)
        image = self.transfrom(img)
        label = tools.cfgch_list.index(self.img_names[idx][0])
        label = torch.from_numpy(np.array(label))
        return image, label
    def evaluate(self, model=None):
        cnt = 0
        total = self.len
        i = total
        while i:
            i -= 1
            image, label = self.__getitem__(i)
            predict = model(image.reshape(1,1,tools.cfg.BG_SIZE,tools.cfg.BG_SIZE))
            index = predict.argmax()
            if index == label: cnt+=1
        print(f'Tested with {total} images.\nPrecision is {100*cnt/total :.2f}%')

    def test_img(self, model=None, img_path=None):
        if img_path:
            img = Image.open(img_path)
            img = np.array(img)
            img = tools.get_binary_img(img)
            img = Image.fromarray(img)
            img = self.transfrom(img)
        else:
            img, _ = self.__getitem__(np.random.randint(self.len))
        predict = model(img.reshape(1,1,tools.cfg.BG_SIZE,tools.cfg.BG_SIZE))
        index = predict.argmax()
        print('Predicted as ' + tools.cfg.ch_list[index])
        image = img.clone()  # clone the tensor
        image = image.squeeze(0)  # remove the fake batch dimension
        image = toPIL(image)
        display(image)
        return image