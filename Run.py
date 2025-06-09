#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 12:08:34 2017

@author: Mohamed
"""

# Append relevant paths
import os
import sys

def conditionalAppend(Dir):
    """ Append dir to sys path """
    if Dir not in sys.path:
        sys.path.append(Dir)

cwd = os.getcwd()
conditionalAppend(cwd)
conditionalAppend(cwd + "/tensorflow_fcn")

import os
from termcolor import colored

import numpy as np
import scipy.io
from PIL import Image
import csv

import ProjectUtils as putils
from FCN8VGG16Model import FCN8VGG16Model
from FCN8VGG16Model_Run import FCN8VGG16Model_Run


#%%============================================================================
# Assign run mode
#==============================================================================

Allowed_RunModes = ["train", "predict_test", "predict_all", "predict_unlabeled"]

if (len(sys.argv) < 3) or (sys.argv[1] not in Allowed_RunModes):
    
    print('Usage: ' + sys.argv[0] + ' <runmode> <path_to_dir_containing_"Run_settings.py">')
    print('Available runmodes: ' + str(Allowed_RunModes))
    
    exit(1)

RUNMODE = sys.argv[1]
#RUNMODE = 'train'

# Import run settings
os.system("touch " + sys.argv[2] + "__init__.py")
conditionalAppend(sys.argv[2])
import Run_settings

# Warn about scalefactor
if (Run_settings.SCALEFACTOR != 1) and \
   (RUNMODE in ['train', 'predict_test', 'predict_all']):
       Msg = colored("\nCAREFUL:\n"+ \
                     "SCALEFACTOR != 1 (labels will be divided by SCALEFACTOR)" + \
                     "\nPress Enter to continue (or CTRL+C to abort) ...", \
                     'yellow')
       input(Msg)

#%%============================================================================
# Instantiate model
#==============================================================================

if RUNMODE not in ["predict_test", "predict_unlabeled"]:
    thisModel = FCN8VGG16Model(**Run_settings.modelparams)
    ModelInfo = thisModel.get_ModelInfo()


#%%============================================================================
# Now run model to: train / predict_test / predict_all / predict_unlabeled
#==============================================================================

if RUNMODE == "train":
    
    #%%========================================================================
    # Train model
    #==========================================================================
    
    thisRun = FCN8VGG16Model_Run(thisModel, RunParams = Run_settings.runparams)
    thisRun.set_GPUOpts(N_GPUs = Run_settings.N_GPUs)
    #RunInfo = thisRun.get_RunInfo()
    thisRun.Run()


elif RUNMODE == "predict_test":
    
    #%%========================================================================
    # Test model on testing set
    #==========================================================================
    
    putils.makeSubdir(Run_settings.modelparams['RESULTPATH'], 'testSet')
    resPath = Run_settings.modelparams['RESULTPATH'] + 'testSet/'
                                    
    Run_settings.modelparams['RESULTPATH'] = resPath
    
    # Instantiate model
    thisModel = FCN8VGG16Model(**Run_settings.modelparams)
    ModelInfo = thisModel.get_ModelInfo()

    Run_settings.runparams['IS_TESTING'] = True
    Run_settings.runparams['PREDICT_ALL'] = False
    Run_settings.runparams['SUBBATCH_SIZE'] = 1         
    Run_settings.runparams['BIGBATCH_SIZE'] = 1
             
    # Run the model
    thisRun = FCN8VGG16Model_Run(thisModel, RunParams = Run_settings.runparams)
    thisRun.set_GPUOpts(N_GPUs = Run_settings.N_GPUs)
    #RunInfo = thisRun.get_RunInfo()
    thisRun.Run()
    
    # Plot comparisons of labels and predictions (testing)
    thisRun.Model.PlotComparisons(SCALEFACTOR = thisRun.SCALEFACTOR)


elif RUNMODE == "predict_all":

    #%%========================================================================
    # Predict the rest of images
    #==========================================================================
    
    Run_settings.runparams['IS_TESTING'] = True
    Run_settings.runparams['PREDICT_ALL'] = True
    Run_settings.runparams['SUBBATCH_SIZE'] = 1         
    Run_settings.runparams['BIGBATCH_SIZE'] = 1

    thisRun = FCN8VGG16Model_Run(thisModel, RunParams = Run_settings.runparams)
    thisRun.set_GPUOpts(N_GPUs = Run_settings.N_GPUs)
    #RunInfo = thisRun.get_RunInfo()
    thisRun.Run()
    
    # Plot confustion matrix for all images for training/valiation
    thisRun.Model.PlotConfusionMat(SCALEFACTOR = thisRun.SCALEFACTOR)
    
    # Plot comparisons of labels and predictions for training/valiation
    thisRun.Model.PlotComparisons(SCALEFACTOR = thisRun.SCALEFACTOR)


elif RUNMODE == "predict_unlabeled":

    #%%========================================================================
    # Predict an unlabeled set of images
    #==========================================================================

    if 'SplitDataParams' in Run_settings.modelparams:
        Run_settings.modelparams['SplitDataParams'].pop('IMAGEPATH', None)
        Run_settings.modelparams['SplitDataParams'].pop('LABELPATH', None)
        Run_settings.modelparams['SplitDataParams'].pop('labelNames', None)

    # Asigură-te că nu există câmpuri inutile în modul unlabeled
    if 'LABELPATH' in Run_settings.modelparams.get('SplitDataParams', {}):
        Run_settings.modelparams['SplitDataParams'].pop('LABELPATH')
    if 'labelNames' in Run_settings.modelparams.get('SplitDataParams', {}):
        Run_settings.modelparams['SplitDataParams'].pop('labelNames')

    # Instantiate model         
    thisModel = FCN8VGG16Model(**Run_settings.modelparams)
    #ModelInfo = thisModel.get_ModelInfo()

    splitparams_thisRun = {'RESULTPATH': Run_settings.RESULTPATH,
                           'IMAGEPATH': Run_settings.IMAGEPATH,
                           'EXT_IMGS': Run_settings.EXT_IMGS,
                           'IS_UNLABELED' : True,
                           'FREE_DIMS' : True,
                           'SAVE_FOVs' : False,
                           'SHIFT_STEP' : 0}
                       
    Run_settings.runparams['SplitDataParams'] = splitparams_thisRun
    
    # Run the model
    thisRun = FCN8VGG16Model_Run(thisModel, RunParams = Run_settings.runparams)
    thisRun.set_GPUOpts(N_GPUs = Run_settings.N_GPUs)
    #RunInfo = thisRun.get_RunInfo()
    thisRun.Run()
    
    # Maintain original naming convention for images that 
    # were not predicted in subparts
    allPreds = os.listdir(Run_settings.RESULTPATH)
    allPreds = [j for j in allPreds if Run_settings.EXT_IMGS in j]
    
    justNames = [j.split('_rowmin')[0] for j in allPreds]
    
    for j in allPreds:
        print("Fixing name for " + j)
        barename = j.split('_rowmin')[0]
        occurences = [o for o in justNames if o == barename]
        if len(occurences) == 1:
            
            oldName = Run_settings.RESULTPATH + j
            newName = Run_settings.RESULTPATH + barename + Run_settings.EXT_IMGS
            
            if ' ' in oldName:
                oldName = oldName.replace(' ', r'\ ')
                newName = newName.replace(' ', r'\ ')
            
            os.system('mv ' + oldName + ' ' + newName)
        
        
    # # Quantify unlabeled predictions: per-class pixel fractions
    # class_labels = [Run_settings.EXCLUDE_LBL[0]] + Run_settings.CLASSLABELS
    # stats = []
    # result_dir = Run_settings.modelparams['RESULTPATH']
    # for fname in sorted(os.listdir(result_dir)):
    #     if not fname.endswith(Run_settings.EXT_IMGS) and not fname.endswith('.mat'):
    #         continue
    #     path = os.path.join(result_dir, fname)
    #     # Load mask array
    #     if fname.endswith('.mat'):
    #         mat = scipy.io.loadmat(path)
    #         arr = next(v for k, v in mat.items() if not k.startswith('__'))
    #         if arr.ndim == 3:
    #             arr = np.argmax(arr, axis=2)
    #     else:
    #         arr = np.array(Image.open(path))
    #     # Compute counts and fractions
    #     counts = np.bincount(arr.flatten(), minlength=len(class_labels))
    #     fracs = counts / arr.size
    #     stats.append([fname] + fracs.tolist())
    # # Write CSV
    # csv_path = os.path.join(result_dir, 'unlabeled_stats.csv')
    # with open(csv_path, 'w', newline='') as f:
    #     writer = csv.writer(f)
    #     header = ['image'] + [f'class_{c}_frac' for c in class_labels]
    #     writer.writerow(header)
    #     writer.writerows(stats)
    # print(f"Saved unlabeled statistics to {csv_path}")
    # # Print top 5 images with highest necrosis fraction (class_3_frac) using csv module
    # csv_path = os.path.join(result_dir, 'unlabeled_stats.csv')
    # necro_list = []
    # with open(csv_path, newline='') as f:
    #     reader = csv.reader(f)
    #     header = next(reader)
    #     idx_frac3 = header.index('class_3_frac')
    #     idx_img    = header.index('image')
    #     for row in reader:
    #         try:
    #             frac3 = float(row[idx_frac3])
    #             necro_list.append((row[idx_img], frac3))
    #         except ValueError:
    #             continue
    # # Sort descending by necrosis fraction
    # necro_list.sort(key=lambda x: x[1], reverse=True)
    # print("Top 5 images with highest necrosis fraction (class_3_frac):")
    # for img_name, frac in necro_list[:5]:
    #     print(f"{img_name}: {frac:.4f}")
        
    
