import numpy as np
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
import torch.nn.functional as F
import utils

# save trained model to a checkpoint for loading later
def save_model(model,arch,epochs,gpu,learnrate,save_dir,output_size):
    train_data,valid_data = get_datasets()
    model.class_to_idx = train_data.class_to_idx
    if (arch == 'resnet18'):
        classifier = model.fc
    else:
        classifier = model.classifier

    checkpoint = {
        'arch':arch,
        'input_size': model.input_size,
        'output_size': output_size,
        'classifier_layers': [each for each in classifier],
        'state_dict': model.state_dict(),
        'optimizer_state_dict':model.optimizer.state_dict(),
        'epoch':epochs,
        'class_to_idx':model.class_to_idx,
    }
    torch.save(checkpoint, save_dir + '/checkpoint.pth')
    return

# Train our model. This is based on training algorithm from the 1st part of the project
def train_model(arch,model,learnrate,epochs,device='cpu'):
    criterion = nn.NLLLoss()
    steps = 0
    running_loss = 0
    print_every = 40

    # Only train the classifier parameters, feature parameters are frozen
    if (arch == "resnet18"):
        model.optimizer = optim.Adam(model.fc.parameters(), lr=learnrate)
    else:
        model.optimizer = optim.Adam(model.classifier.parameters(), lr=learnrate)
    
    model.to(device)

    model.train()
    trainloader, validloader = get_dataloaders()
    print("Begin training\ntrainloader size: {}".format(len(trainloader.dataset)))
    for ep in range(epochs):
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1

            # Move input and label tensors to the GPU
            inputs, labels = inputs.to(device), labels.to(device)

            # DO NOT FORGET THIS NEXT LINE!!
            model.optimizer.zero_grad()

            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            model.optimizer.step()

            # Validations/Checks
            running_loss += loss.item()
            if steps % print_every == 0:
                    # Make sure network is in eval mode for inference
                    model.eval()

                    # Turn off gradients for validation, saves memory and computations
                    with torch.no_grad():
                        test_loss, accuracy = validate(model, validloader, device)

                    print("Epoch: {}/{}.. ".format(ep+1, epochs),
                          "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                          "Validation Loss: {:.3f}.. ".format(test_loss/len(validloader)),
                          "Validation Accuracy: {:.3f}".format(accuracy/len(validloader))
                        )

                    running_loss = 0

                    # Make sure training is back on
                    model.train()
    print("Training complete.") 
    return model

# Based on validation code from Transfer Learning:
def validate(model, imageloader, device = 'cpu'):
    test_loss = 0
    accuracy = 0
    criterion = nn.NLLLoss()
    model.eval()    
    for images, labels in imageloader:

        model,labels,images = model.to(device), labels.to(device), images.to(device)
        
        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        if (device == 'cuda'):
            accuracy += equality.type(torch.cuda.FloatTensor).mean()    
        else:
            accuracy += equality.type(torch.FloatTensor).mean()

    return test_loss, accuracy


def load_model(arch=None, hidden_units=256, output_size=102):
    if (arch == None):
        print("load_model: no architecture specified!")
        return
    else:
        # My understanding is that the number of inputs to our custom classifier layers needs to be identical
        # to the inputs to the pretrained model classifiers, so I'm pulling that from the first "in_features"
        # parameter of each classifier for the various models and setting it to my "input_size" variable before
        # generating my custom classifier
        if (arch == "alexnet"):
            model = models.alexnet(pretrained=True)
            for param in model.parameters():
                param.requires_grad = False
            input_size=model.classifier[1].in_features                    
            model.classifier = get_classifier_layers(input_size=input_size,hidden_units=hidden_units,output_size=output_size)
            model.input_size = model.classifier[0].in_features
        elif (arch == "resnet18"):
            model = models.resnet18(pretrained=True)
            for param in model.parameters():
                param.requires_grad = False
            input_size=model.fc.in_features             
            model.fc = get_classifier_layers(input_size=input_size,hidden_units=hidden_units,output_size=output_size)
            model.input_size = model.fc[0].in_features
        else:
            model = models.vgg13(pretrained=True)
            for param in model.parameters():
                param.requires_grad = False
            input_size=model.classifier[0].in_features                                    
            model.classifier = get_classifier_layers(input_size=input_size,hidden_units=hidden_units,output_size=output_size)
            model.input_size = model.classifier[0].in_features
        
        return model
            

def get_data_dirs(base_dir='flowers'):
    train_dir = base_dir + '/train'
    valid_dir = base_dir + '/valid'
    test_dir = base_dir + '/test'
    return train_dir, valid_dir

def get_datasets():
    train_dir,valid_dir = get_data_dirs()
    train_transforms, valid_transforms = get_transforms()           
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)               
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    return train_data,valid_data
               
def get_dataloaders():
    train_dir, valid_dir = get_data_dirs()
    train_data, valid_data = get_datasets()
    
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)
    
    return trainloader, validloader

def get_transforms():
    means, sds = utils.get_means_sd()
    train_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(means, sds)
    ])

    valid_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(means, sds)
    ])
    return train_transforms, valid_transforms
    

def get_classifier_layers(input_size=25088,hidden_units=256,output_size=102,layer_count=3):
    
    #NOTE: output_size determined by the number of classes being trained for such as 102 for our flower classifier
    
    # create our hidden layers with in_features and out_features evenly spread out between hidden_units and output_size
    # the total number of hidden layers is determined by the "layer_count" parameter
    # the last hidden layer is our final output layer
    input_list = (np.flip(np.linspace(output_size, hidden_units, num=layer_count),axis=0)).astype(int)    

    # add the first input_layer. input_size should ideally be determined from 
    # the output size of the model architecture features (e.g. VGG has 25088 outputs)
    hidden_layers = nn.ModuleList([nn.Linear(int(input_size), int(input_list[0]))])
    # Add the rest of the input layers
    layer_sizes = zip(input_list[:-1], input_list[1:])
    for (h1, h2) in layer_sizes:
        hidden_layers.extend([nn.ReLU()])
        hidden_layers.extend([nn.Dropout(p=0.5)])
        hidden_layers.extend([nn.Linear(int(h1), int(h2))])
    
    # Add our softMax function
    hidden_layers.extend([(nn.LogSoftmax(dim=1))])
    
    # Convert to a nn.Sequential
    # https://discuss.pytorch.org/t/append-for-nn-sequential-or-directly-converting-nn-modulelist-to-nn-sequential/7104
    sequential = nn.Sequential(*hidden_layers)
    
    return sequential