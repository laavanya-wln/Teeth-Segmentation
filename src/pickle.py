
import pickle
import os
from dataset import *
from models import GreyModel

def get_dataset(datafileprefix):
    fname = datafileprefix + '_dataset.dat'
    if os.path.isfile(fname):
        f = open(fname)
        data = pickle.load(f)
        f.close()
        return data
    else:
        data = Dataset('../data/')
        f = open(fname, 'w')
        pickle.dump(gms,f)
        f.close()
        return data
    

def get_grey_models(data,modelfileprefix,train_width):
    fname = modelfileprefix + '_default_'+str(train_width)+'.dat'
    if os.path.isfile(fname):
        f = open(fname)
        gms = pickle.load(f)
        f.close()
        return gms
    else:
        gms = []
        for index,split in enumerate(LeaveOneOutSplitter(data,Dataset.ALL_TRAINING_IMAGES,Dataset.ALL_TEETH)):
            training_images,training_landmarks,_ = split.get_training_set()
            test_image,test_landmark,_ = split.get_test_example()
            gms.append(GreyModel(training_images,training_landmarks,train_width,test_width))
        f = open(fname, 'w')
        pickle.dump(gms,f)
        f.close()
        return gms