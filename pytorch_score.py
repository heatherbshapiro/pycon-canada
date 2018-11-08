# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license.

import torch
import torch.nn as nn
from torchvision import transforms
import json
import base64
from io import BytesIO
from PIL import Image

from azureml.core.model import Model


def preprocess_image(image_file):
    """Preprocess the input image."""
    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_file)
    image = data_transforms(image).float()
    image = torch.tensor(image)
    image = image.unsqueeze(0)
    return image


def base64ToImg(base64ImgString):
    base64Img = base64ImgString.encode('utf-8')
    decoded_img = base64.b64decode(base64Img)
    return BytesIO(decoded_img)


def init():
    global model
    model_path = Model.get_model_path('dogs')
    model = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.eval()


def run(input_data):
    img = base64ToImg(json.loads(input_data)['data'])
    img = preprocess_image(img)

    # get prediction
    output = model(img)

    classes = ['Chihuahua',
             'Japanese_spaniel',
             'Maltese_dog',
             'Pekinese',
             'Shih',
             'Blenheim_spaniel',
             'papillon',
             'toy_terrier',
             'Rhodesian_ridgeback',
             'Afghan_hound',
             'basset',
             'beagle',
             'bloodhound',
             'bluetick',
             'black',
             'Walker_hound',
             'English_foxhound',
             'redbone',
             'borzoi',
             'Irish_wolfhound',
             'Italian_greyhound',
             'whippet',
             'Ibizan_hound',
             'Norwegian_elkhound',
             'otterhound',
             'Saluki',
             'Scottish_deerhound',
             'Weimaraner',
             'Staffordshire_bullterrier',
             'American_Staffordshire_terrier',
             'Bedlington_terrier',
             'Border_terrier',
             'Kerry_blue_terrier',
             'Irish_terrier',
             'Norfolk_terrier',
             'Norwich_terrier',
             'Yorkshire_terrier',
             'wire',
             'Lakeland_terrier',
             'Sealyham_terrier',
             'Airedale',
             'cairn',
             'Australian_terrier',
             'Dandie_Dinmont',
             'Boston_bull',
             'miniature_schnauzer',
             'giant_schnauzer',
             'standard_schnauzer',
             'Scotch_terrier',
             'Tibetan_terrier',
             'silky_terrier',
             'soft',
             'West_Highland_white_terrier',
             'Lhasa',
             'flat',
             'curly',
             'golden_retriever',
             'Labrador_retriever',
             'Chesapeake_Bay_retriever',
             'German_short',
             'vizsla',
             'English_setter',
             'Irish_setter',
             'Gordon_setter',
             'Brittany_spaniel',
             'clumber',
             'English_springer',
             'Welsh_springer_spaniel',
             'cocker_spaniel',
             'Sussex_spaniel',
             'Irish_water_spaniel',
             'kuvasz',
             'schipperke',
             'groenendael',
             'malinois',
             'briard',
             'kelpie',
             'komondor',
             'Old_English_sheepdog',
             'Shetland_sheepdog',
             'collie',
             'Border_collie',
             'Bouvier_des_Flandres',
             'Rottweiler',
             'German_shepherd',
             'Doberman',
             'miniature_pinscher',
             'Greater_Swiss_Mountain_dog',
             'Bernese_mountain_dog',
             'Appenzeller',
             'EntleBucher',
             'boxer',
             'bull_mastiff',
             'Tibetan_mastiff',
             'French_bulldog',
             'Great_Dane',
             'Saint_Bernard',
             'Eskimo_dog',
             'malamute',
             'Siberian_husky',
             'affenpinscher',
             'basenji',
             'pug',
             'Leonberg',
             'Newfoundland',
             'Great_Pyrenees',
             'Samoyed',
             'Pomeranian',
             'chow',
             'keeshond',
             'Brabancon_griffon',
             'Pembroke',
             'Cardigan',
             'toy_poodle',
             'miniature_poodle',
             'standard_poodle',
             'Mexican_hairless',
             'dingo',
             'dhole',
             'African_hunting_dog']
    softmax = nn.Softmax(dim=1)
    pred_probs = softmax(model(img)).detach().numpy()[0]
    index = torch.argmax(output, 1)

    result = json.dumps({"label": classes[index], "probability": str(pred_probs[index])})
    return result