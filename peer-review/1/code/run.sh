#!/bin/bash

# make sure we're in the code directory
cd "$(dirname "$0")"

# Activate environment
conda env list | awk '{print $1}' | grep -qx "env_214_test" || conda env create --name env_214_test -f environment.yaml

# source ~/anaconda3/etc/profile.d/conda.sh
conda activate env_214_test

# Run EDA 
echo "Running EDA ..."
jupyter nbconvert --to notebook --execute --inplace 01_EDA.ipynb

# Run autoencoder training
# Note: This step can be time-consuming and memory-intensive, and should be run separately on a GPU node. The following command is an example of how to submit this job to a GPU node using Slurm.
# cd code
# mkdir -p logs
# sbatch job.sh configs/exp_029_pretrain.yaml

# Run transfer learning
# Note: This step assumes that the autoencoder training has already been completed and the trained model is saved in the expected location.
# Note: This step can be time-consuming and memory-intensive.
echo "Running Transfer Learning ..."
python run_get_embedding.py

# Run transfer learning analysis
# Note: This step can be time-consuming and memory-intensive.
echo "Running Transfer Learning Analysis ..."
jupyter nbconvert --to notebook --execute --inplace 02_transfer_learning.ipynb

# Run feature engineering
echo "Running Feature Engineering ..."
python run_feature_engineering.py

# Run lr&svm tuning
# Note: This step can be time-consuming, so we run it separately and save the results.
# For reproducibility, we also save the selected features ane best model parameters in a git-friendly format (JSON), so that we can reproduce the results without needing to re-run the tuning step.

# echo "Running Logistic Regression and SVM Tuning ..."
# jupyter nbconvert --to notebook --execute --inplace Model_lr_svm.ipynb

# Run Best Logistic Regression
echo "Running Logistic Regression ..."
python run_lr.py

# Run Best SVM
echo "Running SVM ..."
python run_svm.py

# Run XGBoost
echo "Running XGBoost model"
jupyter nbconvert --to notebook --execute --inplace xgboost.ipynb

echo "Running LightGB model"
jupyter nbconvert --to notebook --execute --inplace lightGBM_model.ipynb
jupyter nbconvert --to notebook --execute --inplace feature_importance.ipynb


echo "LightGBM ..."
python lightgbModel.py


# Run stability check to judgement call
echo "Running Stability Check to Judgement Call ..."
jupyter nbconvert --to notebook --execute --inplace stability_judgement_call.ipynb


# Run reality check
echo "Running Reality Check ..."
jupyter nbconvert --to notebook --execute --inplace Model_reality_check.ipynb



# Deactivate environment

conda deactivate