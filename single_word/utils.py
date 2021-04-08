import os, cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

T = transforms.Compose([
    transforms.Resize(128),
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5]) 
])

class CFG():
    def __init__(self, data_path):
        with open(data_path+'3755.txt', 'r', encoding='utf-8') as f:
            self.ch_list = f.readline()
        self.COLOR_MODE = 'L'
        self.BG_SIZE = 128
        self.BG_COLOR = 'black'
        self.BG = np.array(Image.new(self.COLOR_MODE, (self.BG_SIZE,self.BG_SIZE), self.BG_COLOR))

        self.FONT_NUM = 2
        self.FONT_PATHS = [data_path+'simka.ttf', data_path+'simhei.ttf']
        self.FONT_SIZE = 128
        self.FONT_ENCODING = 'unic' #'bg2312'
        self.FONTS = [
            ImageFont.truetype(self.FONT_PATHS[0], self.FONT_SIZE, encoding=self.FONT_ENCODING),
            ImageFont.truetype(self.FONT_PATHS[1], self.FONT_SIZE, encoding=self.FONT_ENCODING)
        ]
        self.TEXT_COLOR = 'white'
        self.TEXT_LOCATION = (0,0)
        self.ROTATE_ANGLES = [-5, -3, -1, 0, 1, 3, 5]
        self.BLACK_THRESHOLD = 40

class Tools():
    def __init__(self, data_path):
        self.cfg = CFG(data_path)

    def get_font(self):
        return self.cfg.FONTS[np.random.randint(self.cfg.FONT_NUM)]

    def make_noise(self, img):
        noise = np.random.randint(0,256,(self.cfg.BG_SIZE,self.cfg.BG_SIZE))
        _, noise = cv2.threshold(noise.astype('uint8'), 240, 255, cv2.THRESH_BINARY)
        img += noise
        return img

    def transform(self, img):
        h, w = img.shape
        m = cv2.getRotationMatrix2D((h*0.5,w*0.5), np.random.choice(self.cfg.ROTATE_ANGLES), 1)
        img = cv2.warpAffine(img, m, (h, w))
        return img

    def get_sample(self, random=True):
        label = np.random.randint(len(self.cfg.ch_list))
        img = Image.fromarray(self.cfg.BG)
        draw = ImageDraw.Draw(img)
        draw.text(self.cfg.TEXT_LOCATION, self.cfg.ch_list[label], font=self.get_font() if random else self.cfg.FONTS[0], fill=self.cfg.TEXT_COLOR)
        img = np.array(img)
        if random:
            img = self.transform(img)
            img = self.make_noise(img)
        return Image.fromarray(img), label

    def get_real_sample(self, random=True, real_dir=None):
        img_names = os.listdir(real_dir)
        img_name = np.random.choice(img_names)
        ch = img_name[0]
        print(ch)
        label = self.cfg.ch_list.index(ch)

        img = Image.open(real_dir+img_name)
        img = np.array(img)
        img = self.get_binary_img(img)
        if random:
            img = self.transform(img)
            img = self.make_noise(img)
        return Image.fromarray(img), label

    def get_binary_img(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.equalizeHist(gray)

        th = gray.min()+self.cfg.BLACK_THRESHOLD
        _, mask = cv2.threshold(gray, th, 255, cv2.THRESH_BINARY_INV)

        img = cv2.resize(mask, (self.cfg.BG_SIZE, self.cfg.BG_SIZE))
        return img

    def get_normal(self, ch, color, bg_color):
        img = Image.new('RGB', (self.cfg.BG_SIZE,self.cfg.BG_SIZE), bg_color)
        draw = ImageDraw.Draw(img)
        draw.text(self.cfg.TEXT_LOCATION, ch, font=self.cfg.FONTS[0], fill=color)
        return img

    def get_score(self, src, ch):
        _, _src = cv2.threshold(src, 200, 1, cv2.THRESH_BINARY)
        norm = self.get_normal(ch, 'white', 'black')
        norm = cv2.cvtColor(np.array(norm), cv2.COLOR_RGB2GRAY)
        _, _norm = cv2.threshold(norm, 200, 1, cv2.THRESH_BINARY)
        _total = _src + _norm
        _match = _src * _norm

        # uncomment the following lines to show temporary imgs in jupyter notebook
        # _, total = cv2.threshold(_total, 0, 255, cv2.THRESH_BINARY)
        # _, match = cv2.threshold(_match, 0, 255, cv2.THRESH_BINARY)
        # display(Image.fromarray(src), Image.fromarray(norm))
        # display(Image.fromarray(total), Image.fromarray(match))
        
        m = _match > 0
        t = _total > 0
        score = m.sum() / t.sum()
        return score

    def evaluate_word(self, img_path, model):
        # get ori
        if type(img_path) is str:
            ori = Image.open(img_path).convert('RGB')
        else:
            ori = img_path
        ori = np.array(ori)
        ori = cv2.resize(ori, (self.cfg.BG_SIZE,self.cfg.BG_SIZE))
        # get scr
        src = self.get_binary_img(ori)
        # get predicted char
        img = T(Image.fromarray(src.copy()))
        predict = model(img.reshape(1,1,self.cfg.BG_SIZE,self.cfg.BG_SIZE))
        index = predict.argmax()
        ch = self.cfg.ch_list[index]
        # get score
        score = self.get_score(src, ch)

        # create normal
        # normal = self.get_normal(ch, 'red', 'white')
        normal = self.get_normal(ch, 'blue', 'white') # RGB to BGR, blue shows like red
        normal = np.array(normal)
        # get visual result
        result = Image.fromarray(cv2.addWeighted(normal, 0.2, ori, 0.8, 0))
        return ch, score, result
