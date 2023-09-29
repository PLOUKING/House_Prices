# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 10:57:02 2023

@author: storm
"""
import pandas as pd
import os
from config.definitions import ROOT_DIR
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pylab as plt

load_path = os.path.join(ROOT_DIR, "house-prices-advanced-regression-techniques")
df_train = pd.read_csv(load_path+"/train.csv")




# model = LGBMRegressor()