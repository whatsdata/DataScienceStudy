import pandas as pd
import numpy as np
import lightgbm as lgb
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
        "num_iterations":trial.suggest_int("num_iterations", 1000, 10000),
       "max_depth":trial.suggest_int("max_depth", 2, 20),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6,1)}
    lgbm = lgb.LGBMRegressor(**param)
    lgbm.fit(X_train, y_train,)
    lgbm_pred = lgbm.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, lgbm_pred))

    return rmse

sampler = TPESampler(seed=42)
study = optuna.create_study(
    study_name = 'lightgbm_parameter_opt',
    direction = 'minimize',
    sampler = sampler,
)

def run():
    study.optimize(objective, n_trials=100)
    tuned_param = study.best_trial.params
    lgbm = lgb.LGBMRegressor(**tuned_param)
    return lgbm

if __name__ == "__main__":
    run()