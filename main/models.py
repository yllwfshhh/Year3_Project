from django.db import models
from django.conf import settings

import os
import numpy as np
import cv2 as cv
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
import torch.nn as nn

# M1 : ct ratio
# M2 : calcification crop
# M3 : ranking prediction
WEIGHTS_FILE_M1 = os.path.join(r'C:\Users\ACER\OneDrive\Desktop\Django\yt_tutorial\ctratio_model.pth')
WEIGHTS_FILE_M2 = os.path.join(r'C:\Users\ACER\OneDrive\Desktop\Django\yt_tutorial\calcification_crop_model.pth')
WEIGHTS_FILE_M3 = os.path.join(r'C:\Users\ACER\OneDrive\Desktop\Django\yt_tutorial\calcification_rank_model.pth')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def load_ratio_model():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)
    num_classes = 3 # 
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(WEIGHTS_FILE_M1, map_location=torch.device(device)))
    model.eval()
    return model

def load_calcification_crop_model():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)
    num_classes = 6 # 
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(WEIGHTS_FILE_M2, map_location=torch.device(device)))
    model.eval()
    return model

def load_calcification_rank_model():
    model = torchvision.models.resnet50(pretrained=True)
    model.fc = nn.Sequential(
                nn.Linear(2048, 128),
                nn.ReLU(inplace=True),
                # num of class
                nn.Linear(128, 2)).to(device)
    model.load_state_dict(torch.load(WEIGHTS_FILE_M3, map_location=torch.device(device)))
    model.eval()
    return model

def model1_predict(tensor_image):
    model = load_ratio_model()
    chest_box =[0,0,0,0]
    heart_box =[0,0,0,0]

    with torch.no_grad():
        output = model(tensor_image)

    boxes = output[0]['boxes'].data.cpu().numpy()
    boxes = boxes.astype(np.int32)
    labels = output[0]['labels'].data.cpu().numpy()
    heart_id = 1
    chest_id = 0
    if(labels[0]==2):
        heart_id = 0
        chest_id = 1

    chest_box[0] = boxes[chest_id][0]
    chest_box[1] = boxes[chest_id][1]
    chest_box[2] = boxes[chest_id][2]
    chest_box[3] = boxes[chest_id][3]

    heart_box[0] = boxes[heart_id][0]
    heart_box[1] = boxes[heart_id][1]
    heart_box[2] = boxes[heart_id][2]
    heart_box[3] = boxes[heart_id][3]

    ct_ratio = round((heart_box[2]-heart_box[0])/(chest_box[2]-chest_box[0]),2)
    return ct_ratio,chest_box,heart_box

def model2_predict(tensor_image,pil_image):
    model = load_calcification_crop_model()
    aortic_box =[0,0,0,0]

    with torch.no_grad():
        output = model(tensor_image)

    boxes = output[0]['boxes'].data.cpu().numpy()
    boxes = boxes.astype(np.int32)

    aortic_box[0] = boxes[0][0]
    aortic_box[1] = boxes[0][1]
    aortic_box[2] = boxes[0][2]
    aortic_box[3] = boxes[0][3]

    crop_aortic(pil_image,aortic_box)
    return aortic_box

def model3_predict(aortic_image):

    tensor_image = np.array(aortic_image)
    tensor_image = cv.resize(tensor_image, (224, 224), interpolation=cv.INTER_AREA)
    tensor_image = tensor_image/255.0
    tensor_image = torch.tensor(tensor_image).float()
    tensor_image = tensor_image.permute(2, 0, 1)  # change the order of dimensions
    tensor_image = tensor_image.unsqueeze(0)  # add a batch dimension

    model = load_calcification_rank_model()
    output = model(tensor_image)
    rank = output.argmax().item()
    return rank

def image_preprocess(pil_image):
        # from PIL to numpy array
        image = np.array(pil_image)
        image = cv.resize(image, (417, 417), interpolation=cv.INTER_AREA)
        image = image/255.0
        image = torch.tensor(image).float()
        image = image.permute(2, 0, 1)  # change the order of dimensions
        image = image.unsqueeze(0)  # add a batch dimension
        return image

def crop_aortic(image,aortic_box):
    aortic_image =image.crop((aortic_box[0], aortic_box[1] , aortic_box[2], aortic_box[3]))
    image_path = os.path.join(settings.MEDIA_ROOT, 'aortic_image.jpg')
    aortic_image.save(image_path)
