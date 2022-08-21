import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os

# read data
abs = os.path.abspath('')
abs.replace('\\model','')
df = pd.read_csv(abs + '\\data\\data_final_0124.csv')
df = pd.get_dummies(df, columns = ['sectors', 'growth_level','major' ,'howmarket'])
#
y = df['export']
X = df.drop(['export', 'problem1','problem2'], axis=1)


#수출하는 기업만 추출
X_export = X[y>0]
y_export = y[y>0]

#검사용
y_noexport = y[y==0]
X_noexport = X[y==0]

X_train, X_validation, y_train, y_validation = train_test_split(X_export,y_export, test_size=0.2)
X_train, X_test, y_train, y_test = train_test_split(X_train,y_train, test_size=0.25)


#implement optuna for tuning
def objective(trial):
    param = {
        "random_state":42,
       'learning_rate' : trial.suggest_loguniform('learning_rate', 0.01, 0.1),
       "n_estimators":trial.suggest_int("n_estimators", 3000,10000),
       "max_depth":trial.suggest_int("max_depth", 4, 16),
       "colsample_bytree":trial.suggest_float("colsample_bytree", 0.6, 1.0),
       "min_child_weight": trial.suggest_int("min_child_weight", 0, 10),
        "subsample": trial.suggest_int("subsample",0.6,1),
   "gamma": trial.suggest_int("gamma",50,150) }
    cat_features = range(X_test.shape[1])
    xg = xgb.XGBRegressor(**param)
    xg.fit(X_train, y_train,)
    xg_pred = xg.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, xg_pred))

    return rmse

sampler = TPESampler(seed=42)
study = optuna.create_study(
    study_name = 'xgboost_parameter_opt',
    direction = 'minimize',
    sampler = sampler,
)

def run():
    study.optimize(objective, n_trials=30)
    tuned_param = study.best_trial.params
    xg = xgb.XGBRegressor(**tuned_param)
    return xg

if __name__ == "__main__":
    run()