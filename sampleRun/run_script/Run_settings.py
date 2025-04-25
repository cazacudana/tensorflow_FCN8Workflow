#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 16:53:31 2017

@author: mohamedt
"""


#%%============================================================================
# Define relevant paths
#==============================================================================

# define relevant paths

import os
cwd = os.getcwd()

print("Running run settings from sample run")
print()
#small data set 
IMAGEPATH = cwd + "/sampleRun/images/trainimagesold/"
#Big data set
#IMAGEPATH = cwd + "/sampleRun/images/"

#small data set 
LABELPATH = cwd + "/sampleRun/images/" + "GTinfo_old/"
#Big data set
#LABELPATH = IMAGEPATH + "GTinfo/"
#LABELPATH = ""

MODELPATH_LOAD = cwd + "/sampleRun/model/"
MODELPATH_SAVE = MODELPATH_LOAD

RESULTPATH = cwd + "/sampleRun/results/"

#%%============================================================================
# Define params
#==============================================================================

#
# single
#

N_GPUs = 1
EXT_IMGS = '.png'
EXT_LBLS = '.png'
SCALEFACTOR = 1

#
# labels and weights
#

CLASSLABELS = [1, 2, 3, 4]

#CLASSWEIGHTS = [1, 1, 1, 1, 1] # Give custom weight
CLASSWEIGHTS = [] # automatically handle class imbalance

cMap = ['red', 'blue' , 'green' , 'yellow']
cMap_lbls = ['tumor', 'stroma','inflamatory infiltrates','necrosis' ]
EXCLUDE_LBL = [0, 5, 6, 7]

#
# Dicts
#

splitparams = {'IMAGEPATH': IMAGEPATH, 
               'LABELPATH': LABELPATH,
               
               'IS_UNLABELED': False,
               'SAVE_FOVs': True,
               
               'PERC_TRAIN' : 0.9, 
               'PERC_TEST' : 0.1, 
               'EXT_IMGS' : EXT_IMGS, 
               'EXT_LBLS' : EXT_LBLS,
               
               'TRAIN_DIMS' : (256, 256),
               'SHIFT_STEP' : 30,
               'IGNORE_THRESH': 0.9,
               'EXCLUDE_LBL': EXCLUDE_LBL,
               'CLASSLABELS': CLASSLABELS,
               'SCALEFACTOR': SCALEFACTOR}


modelparams = {'RESULTPATH' : RESULTPATH, 
               'MODELPATH_LOAD' : MODELPATH_LOAD, 
               'MODELPATH_SAVE' : MODELPATH_SAVE, 
               
               'SplitDataParams' : splitparams, 
               'CLASSLABELS' : CLASSLABELS, 
               'CLASSWEIGHTS' : CLASSWEIGHTS, 
               'cMap' : cMap, 
               'cMap_lbls' : cMap_lbls,
                
               'SplitDataParams' : splitparams, }

splitparams_thisRun = None # existing model-specific set of images

runparams = {'USE_VALID' : True, 
             'IS_TESTING' : False, 
             'PREDICT_ALL' : False, 
             
             'AUGMENT': True,
             'LEARN_RATE' : 1e-6,
             'SUBBATCH_SIZE' : 3,
             'BIGBATCH_SIZE' : 6,
             'MODELSAVE_STEP': 10,
             'MODEL_BACKUP_STEP': 10,
             
             'SCALEFACTOR': SCALEFACTOR,
             't_mins': None,
             'Monitor': True,
             'SplitDataParams': splitparams_thisRun, 
             'USE_MMAP': False}
