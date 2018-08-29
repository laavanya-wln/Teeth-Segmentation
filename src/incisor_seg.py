# -*- coding: utf-8 -*-
"""
Created on Sat May 12 09:38:17 2018

@author: LAAVANYA
"""

import pickle as pickle
import numpy as np
import cv2
from dataset import *
from utils import *
from shape import *
from imgproc import extract_patch_normal, image_transformation_default, patch_transformation_default
from models import PointDistributionModel,GreyModel,ActiveShapeModel,AppearanceModel


#Loading all data  
data = Dataset('../data/')


i = 0
#Leave one out analysis
for index,split in enumerate(LeaveOneOutSplitter(data,Dataset.ALL_TRAINING_IMAGES,Dataset.ALL_TEETH)):
    #Load the training and test data
    training_images,training_landmarks,training_segmentations = split.get_training_set()
    test_image,test_landmark,big_test_segmentation = split.get_test_example()
    transformed_test_image = test_image
    transformed_training_images = training_images
    #Split all landmarks into lower and upper teeth middle teeth
    split_training_landmarks = tooth_splitter(training_landmarks,2) #split into top and bottom
    all_split_training_landmarks = tooth_splitter(training_landmarks,8) #split into individual tooth
    two_upper_teeth = all_split_training_landmarks[1] 
    two_lower_teeth = all_split_training_landmarks[5] 
    ltl=[]
    utl = []
    mtl = []
    for i in range(len(all_split_training_landmarks[0])):
        ashape = all_split_training_landmarks[1][i].concatenate(all_split_training_landmarks[2][i]) #two upper teeth
        bshape = all_split_training_landmarks[5][i].concatenate(all_split_training_landmarks[6][i]) #two lower teeth
        cshape = ashape.concatenate(bshape) #Upper and lower
        utl.append(ashape)
        ltl.append(bshape)
        mtl.append(cshape)
    two_upper_teeth=ShapeList(utl)
    two_lower_teeth=ShapeList(ltl)
    four_middle_teeth = ShapeList(mtl)
    #Point Distribution Model -
    #1. Proscutes Analysis - Normalize shapes
    #2. ModedPCAModel - Dimensionaltiy Reduction
    shape_model = PointDistributionModel(training_landmarks,pca_variance_captured=0.99,use_transformation_matrix=True,project_to_tangent_space=True)
    grey_model = GreyModel(training_images,training_landmarks,20,50,image_transformation_function=image_transformation_default,patch_transformation_function=patch_transformation_default)
    active_shape_model = ActiveShapeModel(shape_model,grey_model)
    appearance_model = AppearanceModel(transformed_training_images,shape_model,[1,1],[1.1,1.5],5)
    appearance_shape,val = appearance_model.fit(transformed_test_image)

    plot_shapes([test_landmark,appearance_shape],labels=['test_landmark','template match initialization'])
    print('Initialization',split.get_dice_error_on_test(centroid_shape),split.get_dice_error_on_test(appearance_shape))
    
    newest_shape1,_,_ = active_shape_model.fit(transformed_test_image,initial_shape=appearance_shape,tol=0.2, max_iters=10)
    newest_shape3,_,_ = active_shape_model.fit(transformed_test_image,initial_shape=test_landmark,tol=0.2, max_iters=10)
    
    plot_shapes([test_landmark,newest_shape1],labels=['test_landmark','asm with template match initialization'])
    plot_shapes([test_landmark,newest_shape2],labels=['test_landmark','asm with split template match initialization'])
    plot_shapes([test_landmark,newest_shape3],labels=['test_landmark','asm with manual initialization'])
    print('ASM',split.get_dice_error_on_test(newest_shape1),split.get_dice_error_on_test(newest_shape2),split.get_dice_error_on_test(newest_shape3)) 
    
    i = i+1
    if i >= 1:
        pass#break

