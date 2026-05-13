import mlflow
import mlflow.sklearn
import argparse
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error

def parse_args():
    parser = argparse.ArgumentParser(description="Training housing price model")
    parser.add_argument("--n-estimators", type=int, default= 100)
    parser.add_argument('--max-depth', type=int, default= 10)
    parser.add_argument('--model-type', type= str, default= 'random_forest')
    return parser.parse_args()

def load_and_prepare_data():
    housing = fetch_california_housing()
    X, y = housing.data, housing.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

def create_model(model_type, n_estimators, max_depth):
    if model_type == 'random_forest':
        return RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
        )
    elif model_type == 'gradient_boosting':
        return GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
        )
    else:
        raise ValueError(f'Model type tidak dikenal: {model_type}')
    
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return{
        "rmse" : root_mean_squared_error(y_test, y_pred),
        'mae'  : mean_absolute_error(y_test, y_pred),
        'r2_score'  : r2_score(y_test,y_pred)
    }

def main():
    args = parse_args()

    print(f'Training {args.model_type} model...')
    print(f'n_estimators: {args.n_estimators}')
    print(f'max_depth: {args.max_depth}')

    X_train, X_test, y_train, y_test = load_and_prepare_data()

    mlflow.set_experiment('housing-project-runs')

    with mlflow.start_run():
        mlflow.log_params({
            'model_type': args.model_type,
            'n_estimators': args.n_estimators,
            'max_depth':args.max_depth,
        })
    
        model = create_model(args.model_type, args.n_estimators, args.max_depth)
        model.fit(X_train,y_train)

        metrics = evaluate_model(model, X_test, y_test)
        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(model,'model')

        print('Training selesai')
        for k, v in metrics.items():
            print(f'{k}: {v:.4f}')
        
if __name__ == '__main__':
    main()