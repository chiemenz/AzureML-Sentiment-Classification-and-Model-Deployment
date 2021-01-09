"""
This script can be used to train and evaluate an XGBoost classifier for hyperparameter tuning with
Azureml hyperdrive
"""
# Import packages

from azureml.core import Dataset, Datastore, Workspace, Experiment
import argparse
import os
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score
import pandas as pd
from azureml.core.run import Run
import joblib


# Here the main method is defined for fitting the classifier and computing its accuracy
run = Run.get_context()


def main():
    # Add arguments to script
    parser = argparse.ArgumentParser(description="hyperparameters of the logistic regression model")
    parser.add_argument('--test-set', type=str, 
                        help="Name of your test set")
    parser.add_argument('--train-set', type=str, 
                        help="Name of your training set")
    parser.add_argument('--max-depth', type=int, default=3,
                        help="How deep is the tree growing during one round of boosting")
    parser.add_argument('--min-child-weight', type=int,
                        default=2,
                        help="Minimum sum of weight for all observations in a child. Controls overfitting")
    parser.add_argument('--gamma', type=float,
                        default=0,
                        help="Gamma corresponds to the minimum loss reduction required to make a split.")
    parser.add_argument('--subsample', type=float,
                        default=0.9,
                        help="What fraction of samples are randomly sampled per tree.")
    parser.add_argument('--colsample-bytree', type=float,
                        default=0.8,
                        help="What fraction of feature columns are randomly sampled per tree.")
    parser.add_argument('--reg-alpha', type=float,
                        default=0.00001,
                        help="L1 regularization of the weights. Increasing the values more strongly prevents "
                             "overfitting.")
    parser.add_argument('--eta', type=float,
                        default=0.2,
                        help="Learning rate for XGBoost.")
    parser.add_argument('--n-estimators', type=int,
                        default=500,
                        help="Number of XGBoost estimators.")
    parser.add_argument('--seed', type=int,
                        default=42,
                        help="Random seed.")

    args = parser.parse_args()
    params = {
        'eta': args.eta,
        'n_estimators': args.n_estimators,
        'max_depth': args.max_depth,
        'min_child_weight': args.min_child_weight,
        'gamma': args.gamma,
        'subsample': args.subsample,
        'colsample_bytree': args.colsample_bytree,
        'reg_alpha': args.reg_alpha,
        'seed': args.seed,
        'objective': 'multi:softmax',
        'num_class': 3,
    }
    
    ws = run.experiment.workspace        
     
    run.log("begin", "Started the run")
    dataset_training = Dataset.get_by_id(ws, id=args.train_set)
    dataset_test = Dataset.get_by_id(ws, id=args.test_set)
    run.log("loaded_dataset", str(dataset_test))

    # Define test and train sets
    train_df = dataset_training.to_pandas_dataframe()
    test_df = dataset_test.to_pandas_dataframe()

    x_train = train_df.drop(columns=['norm_rating']).to_numpy()
    y_train = list(train_df.norm_rating)
    x_test = test_df.drop(columns=['norm_rating']).to_numpy()
    y_test = list(test_df.norm_rating)

    run.log("train_start", "Start training")
    run.log("Loaded a dataset with sample size:", np.int(x_train.shape[0]))
    run.log("Number of estimators:", np.float(args.n_estimators))
    run.log("max depth:",  np.int(args.max_depth))
    run.log("min_child_weight:", np.float(args.min_child_weight))
    run.log("gamma", np.float(args.gamma))
    run.log("subsample:", np.float(args.subsample))
    run.log("colsample_bytree:",  np.float(args.colsample_bytree))
    run.log("reg alpha:", np.float(args.reg_alpha))
    run.log("learning rate:", np.float(args.eta))    

    model = xgb.XGBClassifier(
        objective=params['objective'],
        eta=params['eta'],
        max_depth=params['max_depth'],
        gamma=params['gamma'],
        n_estimators=params['n_estimators'],
        seed=params['seed'],
        colsample_bytree=params['colsample_bytree'],
        reg_alpha=params['reg_alpha'],
        num_class=params['num_class']
    )

    model.fit(x_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(x_test))

    run.log("Accuracy", np.float(accuracy))
    os.makedirs('outputs', exist_ok=True)
    # note file saved in the outputs folder is automatically uploaded into experiment record
    joblib.dump(value=model, filename='outputs/xgboost_model.pkl')


if __name__ == '__main__':
    main()
