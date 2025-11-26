#!/bin/bash
# Install dependencies if needed
/opt/anaconda3/bin/python3 -m pip install pandas numpy scikit-learn xgboost lightgbm catboost

# Run the training script
/opt/anaconda3/bin/python3 main.py
