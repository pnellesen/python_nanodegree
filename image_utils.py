import numpy as np
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
import torch.nn.functional as F
from os import listdir
from PIL import Image
import random
import json
import utils

def predict(cat_file,model=None,image_path=None, topk=1, dvc='cpu'):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    with open(cat_file, 'r') as f:
        cat_to_name = json.load(f)
    
    # Check if cuda is available, use it if so
    if (torch.cuda.is_available() and dvc == 'cuda'):
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    # We can get the pre-processed image using the "get_random_image_from_dir(image_path)" we created earlier
    # if no image_path is provided
    if (image_path == None):
        rand_img_path,img_class = get_random_image_from_dir(cat_to_name)
        rand_img = Image.open(rand_img_path)
    else:
        rand_img = Image.open(image_path)
    
    test_image_tensor=process_image(rand_img)
    
    test_image_tensor.unsqueeze_(0) 
    test_image_tensor=test_image_tensor.float() 
    
    # Similar to I&V lesson:
    if (model != None):
        model.eval()
        model,test_image_tensor = model.to(device), test_image_tensor.to(device)
        
        # Calculate the class probabilities (softmax) for img
        with torch.no_grad():
            output = model.forward(test_image_tensor)

        ps = torch.exp(output)
        probs,indices = ps.topk(topk)

        #https://pytorch.org/docs/stable/tensors.html#torch.Tensor.tolist
        probs = probs.tolist()[0]
        indices = indices.tolist()[0]
        
        classes = []
        for idx in indices:
            
            #https://stackoverflow.com/questions/8023306/get-key-by-value-in-dictionary
            img_class = list(model.class_to_idx.keys())[list(model.class_to_idx.values()).index(idx)]
            
            classes.append(img_class)
        
        top_k_names = []
        for this_class in classes:
            top_k_names.append(cat_to_name[this_class])
        
        return probs, classes, top_k_names
    
    return
    #END predict()

def process_image(image, title=None):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Pytorch Tensor
    '''
    # 1. Resize/crop the image    
    transform_img = transforms.Compose([
        transforms.Resize((256)),
        transforms.CenterCrop(224)
    ])
    img = transform_img(image)
    
    # 2. add conversions
    means_list, sd_list = utils.get_means_sd()
    np_img = ((np.array(img) / 255) - means_list) / sd_list
    np_img_tsp = np_img.transpose((2,0,1))
    
    # 3. now convert numpy array back to a Pytorch tensor.
    output = torch.from_numpy(np_img_tsp)
    
    return output
    # END process_image
    
def get_random_image_from_dir(cat_to_name,img_dir='flowers/test'):
    # Get a random image from our test folder if no image directory is specified
    rand_img_folder = str(random.randint(1,len(listdir(img_dir))-1))
    img_class_name = cat_to_name[str(rand_img_folder)]
    rand_img_path = img_dir + "/" + rand_img_folder
    img_dir_list = listdir(rand_img_path)
    rand_img_path += "/" + img_dir_list[random.randint(1,len(img_dir_list)-1)]
    return rand_img_path, img_class_name
    #END get_random_image