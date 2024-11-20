"""
@author: ellenbet
adapted from 
https://stackoverflow.com/questions/76049457/how-to-use-dino-as-a-feature-extractor
"""

import torch
torch.manual_seed(17)
import pickle
from torchvision import transforms as pth_transforms
from PIL import Image 
import os
#from my_utils import extract_name, image_resize


def main(input_pth):
    model_dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    model_dino.eval()
    print(model_dino)

    # testing if we have gpu available:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model_dino.to(device)

    # some standard reccomended settings
    transform = pth_transforms.Compose([
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
      
    input_saveas = input_pth + "_" + "features" # not used yet
    imfls = os.listdir(input_pth)
    testrange = 5

    for n in range(testrange):
        im = imfls[n]
        #for im in imfls:
        img = Image.open(input_pth + "/" + im) # only for black n white: .convert('RGB')
        # img = img.resize((1400, 1036), Image.LANCZOS) not essential yet
        # TODO - some function that extract the name we want to save the features as 
        img_tensor = transform(img)	
        img_tensor = img_tensor.unsqueeze(0).cuda()		

        with torch.no_grad():
            feats = model_dino(img_tensor)

        # converts features from tensor objects to arrays
        feats = feats.cpu().numpy()[0]
        # TODO something that saves our data

        # viewing features to see how they look
        print("RUN", n, "FEATURES: ", feats)
        print("FEATURE LENGTH:", len(feats))

        # Consider - pickle, but perhaps csv is better
        #with open(treatment + input_saveas + '.txt', 'wb') as handle:
        #    pickle.dump(plate, handle, protocol = pickle.HIGHEST_PROTOCOL)
