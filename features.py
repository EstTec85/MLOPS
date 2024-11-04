#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 17:28:12 2024

@author: estebanjimenez
"""

import pickle

features_path = '/Users/estebanjimenez/Library/CloudStorage/OneDrive-Personal/Tec_Monterrey/Maestria_Aya/MLOPS/ML_OPS_WINE/selected_features.pkl'

with open(features_path, "rb") as f:
    selected_features = pickle.load(f)

print("Características seleccionadas:", selected_features)
