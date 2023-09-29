# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 10:57:02 2023

@author: storm
"""
import pandas as pd
import os
from config.definitions import ROOT_DIR
load_path = os.path.join(ROOT_DIR, "house-prices-advanced-regression-techniques")
train = pd.read_csv(load_path+"/train.csv")





from lightgbm import LGBMClassifier
model = LGBMClassifier()