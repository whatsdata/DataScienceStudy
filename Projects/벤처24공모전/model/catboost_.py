import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os

# read data
abs = os.path.abspath('')
abs.replace('\\model','')
df = pd.read_csv(abs + '\\data\\data_final_0124.csv')

#X,y 분해
y = df['export']
X = df.drop(['export', 'problem1',
       'problem2'], axis=1)
#애로사항은 결과 비교에 사용

#수출하는 기업만 추출
X_export = X[y>0]
y_export = y[y>0]

#검사용
y_noexport = y[y==0]
X_noexport = X[y==0]

X_train, X_validation, y_train, y_validation = train_test_split(X_export,y_export, test_size=0.2)
X_train, X_test, y_train, y_test = train_test_split(X_train,y_train, test_size=0.25)

# extract cat features
cat = CatBoostRegressor()
cat_features = ['sectors', 'growth_level','major' ,'howmarket']
a = [np.where(X_train.columns ==cat_features[i]) for i in range(len(cat_features))]
cat_features = []
for i in range(len(a)):
    cat_features.append(int(a[i][0][0]))
cat_features = np.array(cat_features).astype('int64')

def objective(trial):
    param = {
      "random_state":42,
      'learning_rate' : trial.suggest_loguniform('learning_rate', 0.01, 0.05),
      'bagging_temperature' :trial.suggest_loguniform('bagging_temperature', 0.01, 100.00),
      "n_estimators":trial.suggest_int("n_estimators", 500, 5000),
      "max_depth":trial.suggest_int("max_depth", 4, 16),
      'random_strength' :trial.suggest_int('random_strength', 0, 100),
      "colsample_bylevel":trial.suggest_float("colsample_bylevel", 0.4, 1.0),
      "l2_leaf_reg":trial.suggest_float("l2_leaf_reg",1e-8,3e-5),
      "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
      "max_bin": trial.suggest_int("max_bin", 200, 500),
      'od_type': trial.suggest_categorical('od_type', ['IncToDec', 'Iter']),
  }
    cat = CatBoostRegressor(**param)
    cat.fit(X_train, y_train,
          eval_set=[(X_train, y_train), (X_test,y_test)],
          early_stopping_rounds=35,cat_features=cat_features,
          verbose=100)
    cat_pred = cat.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, cat_pred))
    return rmse

sampler = TPESampler(seed=42)
study = optuna.create_study(
    study_name = 'cat_parameter_opt',
    direction = 'minimize',
    sampler = sampler,
)

def run():
    study.optimize(objective, n_trials=100)
    tuned_params = study.best_trial.params
    cat = CatBoostRegressor(**tuned_params)
    return cat

if __name__ == "__main__":
    run()