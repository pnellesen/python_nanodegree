import argparse
import torch
from torch import nn
from torchvision import datasets, transforms, models
import torch.nn.functional as F

# set up our means and standard deviations for our calculations
def get_means_sd():
    means_list = [0.485, 0.456, 0.406]
    sd_list = [0.229, 0.224, 0.225]
    return means_list, sd_list

# load a saved checkpoint
def load_checkpoint(filepath,dvc):
    if (dvc == 'gpu'):
        device = torch.device("cuda")
        checkpoint = torch.load(filepath)
    else:
        checkpoint = torch.load(filepath, map_location='cpu')
    
    #https://discuss.pytorch.org/t/append-for-nn-sequential-or-directly-converting-nn-modulelist-to-nn-sequential/7104
    classifier_layers = nn.Sequential(*checkpoint['classifier_layers'])
     
    #"Loading the state dict works only if the model architecture is exactly the same as the checkpoint architecture"
    # Since we originally used one of the torchvision.models as our model, we can load it back here, then replace the classifier
    # and state_dict with our saved version
    if (checkpoint['arch'] == 'vgg13'):
        model = models.vgg13(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.classifier = classifier_layers
        
    elif (checkpoint['arch'] == 'resnet18'):
        model = models.resnet18(pretrained=True)
        for param in model.parameters():
                param.requires_grad = False
        model.fc = classifier_layers
    else:
        model = models.alexnet(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.classifier = classifier_layers

    model.load_state_dict(checkpoint['state_dict'])    
    model.class_to_idx = checkpoint['class_to_idx']

    return model

# get and parse our arguments for predict.py 
def get_predict_args():
   
    # create the argparser
    parser = argparse.ArgumentParser()
    
    # setup default values
    default_image_dir = 'flowers/'
    default_save_dir = 'savefiles/'
    default_cat_names = 'cat_to_name.json'
    default_top_k = 1
    gpu = False
    
    #add arguments to parser
    parser.add_argument('image_path', type=str, help='Path to image files. Required (Base image directory: {}'.format(default_image_dir))
    parser.add_argument('checkpoint', type=str, help='Checkpoint to load. Required (Checkpoints saved in: {}'.format(default_save_dir))
    parser.add_argument('--top_k', type=int, default=default_top_k, help='Number of most probable classes/names to print. Default: 1')
    parser.add_argument('--category_names', type=str, default=default_cat_names, help='File to use for mapping flower categories to actual names. Default: {}'.format(default_cat_names))
    parser.add_argument('-gpu', '--gpu', action='store_true', help="Use GPU to train then network")
     
    return parser.parse_args()


# get and parse our command line arguments for train.py
def get_model_args():
   
    # create the argparser
    parser = argparse.ArgumentParser()
    
    # setup default values
    arch_choices = ['vgg13','alexnet', 'resnet18']
    default_arch = arch_choices[0]
    default_learn_rate = 0.001
    default_hidden_units = 512
    default_epochs = 3
    default_data_dir = 'flowers/'
    default_save_dir = 'savefiles/'
    gpu = False
    # number of classes in our data_dir (eg: '/flowers'). Could possibly calculate this using python filesystem functions
    default_output_size = 102
    
    #add arguments to parser
    parser.add_argument('--learn_rate', type=float, default=default_learn_rate, help='Set the learning rate of the network. Default: {}'.format(default_learn_rate))
    parser.add_argument('--epochs', type=int, default=default_epochs, help='Number of epoch for training. Default: {}'.format(default_epochs))
    parser.add_argument('--hidden_units', type=int, default=default_hidden_units, help='Number of starting out_values to the classifier layer. Default: {}'.format(default_hidden_units))
    parser.add_argument('--data_dir', type=str, default=default_data_dir, help='Path to image files. Default: {}'.format(default_data_dir))
    parser.add_argument('--save_dir', type=str, default=default_save_dir, help='Checkpoint save directory. Default: {}'.format(default_save_dir))
    parser.add_argument('--arch', type=str, default=default_arch, choices=arch_choices, help='Pretrained model architecture to use for image classification. Select from one of the following: {} (default is {})'.format(arch_choices, default_arch))
    parser.add_argument('-gpu', '--gpu', action='store_true', help="Use GPU to train then network")
    parser.add_argument('--output_size', type=str, default=default_output_size, help="Number of outputs/classes from our classifier. Default: {}".format(default_output_size))
    
    return parser.parse_args()