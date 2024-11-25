"""
@author: ellenbet
adapted from 
https://stackoverflow.com/questions/76049457/how-to-use-dino-as-a-feature-extractor


USED TO RETRIEVE features_padded_plankton
This works. Input in main is outer dir, assumption is that there exists an inner dictionary. 
"""

import torch
torch.manual_seed(17)
import pickle
from torchvision import transforms as pth_transforms
from PIL import Image 
import os
import numpy as np
#from my_utils import extract_name, image_resize

def savefeatures(outerdir, innerdir, image_name, feature_array):
    full_fname = outerdir + "/" + innerdir + "/" + image_name
    try:
        np.savetxt("features_" + full_fname.replace(".jpg", ".txt") , feature_array)
    except FileNotFoundError:
        os.mkdir("features_" + outerdir + "/" + innerdir, 0o755 );
        np.savetxt("features_" + full_fname.replace(".jpg", ".txt") , feature_array)


def main(outer_dir):
    try:
        os.mkdir("features_" + outer_dir, 0o755)
    except FileExistsError:
        pass
    
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
    
    inner_dirs = os.listdir(outer_dir)

    for inner_dir in inner_dirs: 
        for im in [image for image in os.listdir(outer_dir + "/" + inner_dir) if not image.startswith(".")]:
            full_path = outer_dir + "/" + inner_dir + "/" + im
            img = Image.open(full_path) # only for black n white: .convert('RGB')
            img = img.resize((1400, 1036), Image.LANCZOS)
            img_tensor = transform(img)	
            img_tensor = img_tensor.unsqueeze(0).cuda()		

            with torch.no_grad():
                feats = model_dino(img_tensor)

            # converts features from tensor objects to arrays
            feats = feats.cpu().numpy()[0]        
            savefeatures(outer_dir, inner_dir, im, feats)

main("padded_plancton")