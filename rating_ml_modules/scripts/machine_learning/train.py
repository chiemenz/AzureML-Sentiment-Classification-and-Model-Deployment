"""
This script can be used to train and evaluate an XGBoost classifier for hyperparameter tuning with
Azureml hyperdrive
"""
import argparse
import os

import joblib
import numpy as np
import scipy.sparse
import xgboost as xgb
from azureml.core import Dataset
from azureml.core.run import Run
from sklearn.metrics import f1_score, accuracy_score

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
    parser.add_argument('--seed', type=int,
                        default=42,
                        help="Random seed.")
    parser.add_argument('--num-iterations', type=int,
                        default=20,
                        help="Number of fitting iterations")

    args = parser.parse_args()

    params = {
        'eta': args.eta,
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

    run.log("max depth:", np.int(args.max_depth))
    run.log("min_child_weight:", np.float(args.min_child_weight))
    run.log("gamma", np.float(args.gamma))
    run.log("subsample:", np.float(args.subsample))
    run.log("colsample_bytree:", np.float(args.colsample_bytree))
    run.log("reg alpha:", np.float(args.reg_alpha))
    run.log("learning rate:", np.float(args.eta))

    # Load the Training Dataset and the Test Dataset
    ws = run.experiment.workspace
    dataset_training = Dataset.get_by_id(ws, id=args.train_set)
    dataset_test = Dataset.get_by_id(ws, id=args.test_set)
    run.log("loaded_dataset", str(dataset_test))

    train_df = dataset_training.to_pandas_dataframe()
    test_df = dataset_test.to_pandas_dataframe()

    # Convert the training and test sets to sparse matrices and then create a xgboost DMatrix for efficient computation
    x_train = scipy.sparse.csr_matrix(train_df.drop(columns=['norm_rating']).to_numpy())
    y_train = list(train_df.norm_rating)
    x_test = scipy.sparse.csr_matrix(test_df.drop(columns=['norm_rating']).to_numpy())
    y_test = list(test_df.norm_rating)

    dtrain = xgb.DMatrix(x_train, label=y_train)
    dtest = xgb.DMatrix(x_test, label=y_test)

    # Train the model
    watchlist = [(dtest, 'eval'), (dtrain, 'train')]
    num_round = args.num_iterations
    model = xgb.train(params, dtrain, num_round, watchlist)

    # Evaluate the model
    pred_test = model.predict(dtest)
    pred_train = model.predict(dtrain)

    # Compute F1 scores and Accuracy scores
    f1_score_weighted_train = f1_score(y_train, pred_train, average='weighted')
    accuracy_train = accuracy_score(y_train, pred_train)
    f1_score_weighted = f1_score(y_test, pred_test, average='weighted')
    accuracy = accuracy_score(y_test, pred_test)

    os.makedirs('outputs', exist_ok=True)
    # note file saved in the outputs folder is automatically uploaded into experiment record
    joblib.dump(value=model, filename='outputs/xgboost_model.pkl')

    run.log("F1ScoreWeightedTrain", np.float(f1_score_weighted_train))
    run.log("F1ScoreWeighted", np.float(f1_score_weighted))
    run.log("AccuracyTrain", np.float(accuracy_train))
    run.log("Accuracy", np.float(accuracy))


if __name__ == '__main__':
    main()
