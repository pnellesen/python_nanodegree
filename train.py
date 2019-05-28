#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND/home/workspace/ImageClassifier
#                                                                             
# TODO: 0. Fill in your information in the programming header below
# PROGRAMMER: Pat Nellesen
# DATE CREATED: 05/20/2019
# REVISED DATE: 05/27/2019            <=(Date Revised - if any)

# PURPOSE:  train a new network on a dataset and save the model as a checkpoint.

import utils, model_utils

def main():
    in_args = utils.get_model_args()
        
    # load the model from the --arch and --hidden_units parameters
    print("Loading the model...")
    model = model_utils.load_model(in_args.arch, in_args.hidden_units)

    # train the model
    print("Now training the model...")
    model = model_utils.train_model(in_args.arch, model, in_args.learn_rate,in_args.epochs,device='cuda' if in_args.gpu else 'cpu')
    
    #save the model
    print("Saving the model to {}".format(in_args.save_dir))
    model_utils.save_model(model,in_args.arch,in_args.epochs,in_args.gpu,in_args.learn_rate,in_args.save_dir,in_args.output_size)
    
if __name__ == "__main__":
    main()    
          