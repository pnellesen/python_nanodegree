#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND/home/workspace/ImageClassifier
#                                                                             
# TODO: 0. Fill in your information in the programming header below
# PROGRAMMER: Pat Nellesen
# DATE CREATED: 05/27/2019
# REVISED DATE:             <=(Date Revised - if any)

# PURPOSE:  Load a saved checkpoint and path to a flower image, and predict the flower name along with the probability of that name. 
import utils, image_utils

def main():
    in_args = utils.get_predict_args()
    model=utils.load_checkpoint(in_args.checkpoint,dvc='cuda' if in_args.gpu else 'cpu')
    probs,classes,names = image_utils.predict(in_args.category_names,model=model,image_path=in_args.image_path,topk=in_args.top_k)
    print("\nPredictions completed")
    print("\tClasses: {}".format(classes))
    print("\tProbabilities: {}".format([round(prob,3) for prob in probs]))
    print("\tMost probable flower names: {}".format(names))
    
if __name__ == "__main__":
    main()      
