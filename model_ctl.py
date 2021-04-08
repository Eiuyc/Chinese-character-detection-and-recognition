import torch, cv2, os, numpy as np
from torchvision import transforms
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords, apply_classifier
from utils.plots import plot_one_box

from single_word.utils import Tools
tools = Tools('single_word/data/')

class Young():
    def __init__(self, yolo_weight_path='./single_word/weights/yolov5s.pt', cnn_weight_path='./single_word/weights/cnn.pt'):
        self.device = torch.device('cpu')
        self.yolo = torch.load(yolo_weight_path, map_location=self.device)['model'].float().fuse().eval()
        self.classifier = torch.load(cnn_weight_path, map_location=self.device).eval()
        # self.classifier_t = transforms.Compose([
        #     transforms.Resize((32,32)),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[.5], std=[.5])
        # ])
        self.conf_thres = 0.25
        self.iou_thres = 0.1
    
    def process_input(self, img0):
        img = letterbox(img0, new_shape=640)[0]
        img = img[:,:,::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(torch.device('cpu'))
        img = img.float()
        img /= 255
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img

    def evaluate(self, img0):
        img_tensor = self.process_input(img0)
        pred = self.yolo(img_tensor)[0]
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, agnostic=True)
        
        pred_result = []
        im0 = img0
        # Process detections
        for det in pred:  # detections per image
            if not len(det): continue
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], im0.shape).round()

            # Write results
            for *xyxy, conf, cls in reversed(det):
                c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                word = im0[c1[1]:c2[1], c1[0]:c2[0]] # y x
                ch, score, result = tools.evaluate_word(word, self.classifier)
                pred_result.append((torch.Tensor(xyxy).numpy().tolist(), conf.numpy().tolist(), ch, score))
                result = np.array(result)
                result = cv2.resize(result, (abs(c1[0] - c2[0]), abs(c1[1] - c2[1]))) # w h
                im0[c1[1]:c2[1], c1[0]:c2[0]] = result
                
                label = f'{conf:.2f} {score*100:.2f}'
                plot_one_box(xyxy, im0, label=label, color=(0,0,255), line_thickness=1)

        return pred_result, im0