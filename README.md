<p align="center">
  <br>
  <a href="">
    <img src=".github/logo.png" width="240"/>
  </a>
</p>

<h1 align="center">
MLUTIL for Day-to-Day <code>ML</code> Workflow</h1>

<p align="center">
📦 Functions for improvising your machine learning workflow even smoother and efficient.  
</p>

<p align="center">

<img src="https://img.shields.io/badge/Python%20%20%F0%9F%90%8D-3.8-blueviolet">
<img src="https://img.shields.io/badge/Machine%E2%9A%9B%EF%B8%8F-Learning-43BFF1">
<img src="https://badgen.net/badge/Open%20Source%20%3F/Yes%21/blue?icon=github">
<img src="https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat">
<a href="https://marketplace.visualstudio.com/items?itemName=Aashish.emoji-in-git-commit">
<img src="https://img.shields.io/badge/Emoji%20%F0%9F%98%9C-Commits-yellow">
</a>
<br>
<br>
</p>

## Getting Started

<!-- Not added to package registry -->
<!-- ### Installation -->
<!-- Install 'mlutil` in your project. -->

<!-- ```sh -->
<!-- pip install mlutil -->
<!-- ``` -->

### Import 

Import `mlutil` functions in your project.

```sh
from mlutil import *
```


## Some Methods + Examples


## 1. 🦄 hyp_pipeline for Hyper Parameters search

This simplifies the entire Hyper-parameters search process in one simple method.
<table width='100%' align="center">
  <tr align='center'>
    <td> XGBoost </td>
    <td> LightGBM </td>
  </tr>
  <tr>
   <td>

```python
model = xgb.XGBRegressor()
# params
param_grid = {
    'n_estimators': [400, 700, 1000],
    'colsample_bytree': [0.7, 0.8],
    'max_depth': [15,20,25],
    'reg_alpha': [1.1, 1.2, 1.3],
    'reg_lambda': [1.1, 1.2, 1.3],
    'subsample': [0.7, 0.8, 0.9]
}

model, pred = hyp_pipeline(X_train, X_test, 
y_train, y_test
, model,param_grid, cv=5)

# Root Mean Squared Error
print(np.sqrt(-model.best_score_))
print(model.best_params_)
``` 
</td>
 <td>

```python
model = lgb.LGBMClassifier()
param_grid = {
    'n_estimators': [400, 700, 1000],
    'colsample_bytree': [0.7, 0.8],
    'max_depth': [15,20,25],
    'num_leaves': [50, 100, 200],
    'reg_alpha': [1.1, 1.2, 1.3],
    'reg_lambda': [1.1, 1.2, 1.3],
    'min_split_gain': [0.3, 0.4],
    'subsample': [0.7, 0.8, 0.9],
    'subsample_freq': [20]
}

model, pred = hyp_pipeline(X_train, X_test, 
y_train, y_test, 
model,param_grid, cv=5, 
scoring_fit='accuracy')

print(model.best_score_)
print(model.best_params_)
```
</td></tr></tr>
<tr align='center'>
<td> Keras </td>
<td> RandomForest</td>
<td> CatBoost</td>
</tr>
<td>

```python
param_grid = {
              'epochs':[1,2,3],
              'batch_size':[128]
              #'epochs' : [100,150,200],
              #'batch_size' : [32, 128],
              #'optimizer' : ['Adam', 'Nadam'],
              #'dropout_rate' : [0.2, 0.3],
              #'activation' : ['relu', 'elu']
             }

model = KerasClassifier(build_fn=build_cnn,verbose=0)

model, pred = hyp_pipeline(X_train, X_test,
 y_train, y_test, model,param_grid,
 cv=5, scoring_fit='neg_log_loss')

print(model.best_score_)
# If you get -negative score
 try to change y values to categorical
# Categorical y values
# y_train = to_categorical(y_train)
# y_test= to_categorical(y_test)
print(model.best_params_)
```
</td>
<td>

```python
model = RandomForestClassifier()
param_grid = {
    'n_estimators': [400, 700, 1000],
    'max_depth': [15,20,25],
    'max_leaf_nodes': [50, 100, 200]
}
model, pred = hyp_pipeline(X_train, X_test,
 y_train, y_test, 
model,param_grid, cv=5, 
scoring_fit='accuracy')

print(model.best_score_)
print(model.best_params_)
```
</td>
<td>

```python
from catboost import Pool, CatBoostClassifier
clf = CatBoostClassifier(iterations=759,
    learning_rate=0.06,
    random_strength=0.1,
    depth=10,
    loss_function='MultiClass',
    eval_metric='Accuracy',
    leaf_estimation_method='Newton',
    task_type="GPU",
    devices='0:1'
)

clf.fit(
    X_train, y_train,
    cat_features=cat_features,
    eval_set=(X_val, y_val),
    verbose=False,
    plot=True
)
clf.score(X_val, y_val)
```
</td>
</tr>


</table>

## 2. 🦄 plot_history for Keras Model history
  plot training loss vs validation loss and training accuracy vs validation accuracy using just one line. 


<table>
<tr>
<td>

```python
history = model.fit(x_train, 
                    y_train,
                    epochs = 200,
                    batch_size = 16,
                    verbose=1,
                    validation_data=(x_val,y_val))
plot_history(history)
```
</td></tr>
<tr><td>
<img src="https://user-images.githubusercontent.com/29516182/93336177-09dafd80-f845-11ea-9852-9c581272d98e.png">
</td></tr>
</table>



## [Changelog →](CHANGELOG.md)

All notable changes in this project's released versions are documented in the [changelog](CHANGELOG.md) file.

## Todo 

*mlutil consists of machine learning methods/short hand techniques that we use on every machine Learning Project for better results.
Methods that are useful for Data mining, Data visuz, Model Evaluation, Improving model parameters and many more.* 

* Select one useful ML workflow.  
* Add your code snippet to [mlutil.py](https://github.com/pawarashish564/mlutil/blob/master/mlutil.py)
* Create Pull Request. 
<!-- ## License -->

MIT ⓒ Aashish Pawar. Follow me on Github [@pawarashish564 →](https://github.com/pawarashish564)
