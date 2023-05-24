from django.shortcuts import render
from django.conf import settings
from .models import *
import os
import torch
import cv2 as cv
from PIL import Image
from django import forms

def index(request):
    return render(request, 'index.html')

def test(request):
    return render(request, 'test.html')

def predict(request):
    if request.method == 'POST':
        # Get the uploaded image from the request
        # if request.FILES['image']:
        #     uploaded_image = request.FILES['image']
        #     image_path = os.path.join(settings.MEDIA_ROOT, uploaded_image.name)
            
        #     with open(image_path, 'wb') as f:
        #         f.write(uploaded_image.read())

        # else:
        image_path = '../static/sample01.jpg'

        # Load the saved image using PIL
        image = Image.open(image_path)
        tensor_image = image_preprocess(image)
   
        ct_ratio, chest_box, heart_box = model1_predict(tensor_image)
        aortic_box = model2_predict(tensor_image,image)

        aortic_image = Image.open(os.path.join(settings.MEDIA_ROOT, "aortic_image.jpg"))
        rank = model3_predict(aortic_image)


        show_image_url = chest_heart_border(image,chest_box,heart_box,aortic_box)
        context = {
            'uploaded_image_url': request.build_absolute_uri(settings.MEDIA_URL + uploaded_image.name),
            'show_image_url' : request.build_absolute_uri(settings.MEDIA_URL + "ct_ratio/show_image.jpg"),
            'predicted_ratio': ct_ratio,
            'rank': rank,

        }
        return render(request, 'predict.html', context)
    return render(request, 'predict.html')

def chest_heart_border(pil_image,chest_box,heart_box,aortic_box):
    image = np.array(pil_image)
    show_image = cv.resize(image, (417, 417), interpolation=cv.INTER_AREA)
    show_image = cv.line(show_image, (chest_box[0],chest_box[3]-20), (chest_box[2],chest_box[3]-20), (255, 0, 0), 2)
    show_image = cv.line(show_image, (chest_box[0],chest_box[3]-25), (chest_box[0],chest_box[3]-15), (255, 0, 0), 2)
    show_image = cv.line(show_image, (chest_box[2],chest_box[3]-25), (chest_box[2],chest_box[3]-15), (255, 0, 0), 2)

    show_image = cv.line(show_image, (heart_box[0],heart_box[3]-35), (heart_box[2],heart_box[3]-35), (0, 0, 255), 2)
    show_image = cv.line(show_image, (heart_box[0],heart_box[3]-40), (heart_box[0],heart_box[3]-30), (0, 0, 255), 2)
    show_image = cv.line(show_image, (heart_box[2],heart_box[3]-40), (heart_box[2],heart_box[3]-30), (0, 0, 255), 2)

    show_image = cv.rectangle(show_image, (aortic_box[0],aortic_box[1]), (aortic_box[2],aortic_box[3]), (0, 255, 0), 2)
    image_path = os.path.join(settings.MEDIA_ROOT, 'ct_ratio', 'show_image.jpg')
    cv.imwrite(image_path, show_image)
    return image_path




