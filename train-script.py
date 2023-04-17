#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, roc_curve 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA


# In[2]:


data = pd.read_csv('/kaggle/input/dataset/dataset.csv')


# In[3]:


np.random.seed(42)


# ## Анализируем

# In[4]:


data.info() # данных очень мало, а признаков много


# In[5]:


np.sum(np.sum(data.isna())) # пропусков нет 


# In[6]:


data.describe()


# In[7]:


for i in range(805): # нет категориальных признаков
    if len(data[str(i)].value_counts()) < 3000:
        print(str(i))


# In[8]:


fig, ax = plt.subplots() # несбалансированный таргет 
data.groupby('target').size().plot(kind='pie', ax=ax)
plt.show()


# ## Готовимся к обучению

# In[9]:


X_train, X_val, y_train, y_val = train_test_split(data.drop(columns = ['target']), data['target'], stratify=data['target'], train_size=0.8, random_state=42)


# In[10]:


def paint_res(model, X_train, X_val, y_train, y_val):
    y_val_predicted = model.predict_proba(X_val)[:, 1]
    y_train_predicted = model.predict_proba(X_train)[:, 1]
    
    train_auc = roc_auc_score(y_train, y_train_predicted)
    test_auc = roc_auc_score(y_val, y_val_predicted)

    plt.figure(figsize=(10,7))
    plt.plot(*roc_curve(y_train, y_train_predicted)[:2], label='train AUC={:.4f}'.format(train_auc))
    plt.plot(*roc_curve(y_val, y_val_predicted)[:2], label='val AUC={:.4f}'.format(test_auc))
    legend_box = plt.legend(fontsize='large', framealpha=1).get_frame()
    legend_box.set_facecolor("white")
    legend_box.set_edgecolor("black")
    plt.plot(np.linspace(0,1,100), np.linspace(0,1,100))
    plt.show()


# ## Базовые модели

# In[12]:


log_pipeline = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=10000, penalty = 'l2')
)

parameters = {
    'logisticregression__C'       :  np.logspace(-4, 4, num=9)
}

log_model = GridSearchCV(estimator=log_pipeline, 
                        param_grid=parameters,
                        cv=5, 
                        scoring='roc_auc',
                        verbose=10)
log_model.fit(X_train, y_train)


# In[13]:


paint_res(log_model, X_train, X_val, y_train, y_val)


# In[14]:


cat_model = CatBoostClassifier()
cat_model.fit(X_train, y_train)


# In[15]:


paint_res(cat_model, X_train, X_val, y_train, y_val)


# ## Рассуждения почему такая модель

# Сделала ставку на CatBoost. Теперь хочется уменьшить количество признаков --- PCA. Без новых признаков, повысить метрику не получалось, поэтому их нужно добавить. В итоге каркас финальной модели будет выглядеть как-то так:
# 
# ? -- то, что будет подбираться GridSearchCV, а остальные параметры были выбраны из других обученных моделей и из вычислительных возможностей компьютера
# 
# ```
# make_pipeline(
#     StandardScaler(),
#     PCA(n_components=3),
#     PolynomialFeatures(degree=3),
#     CatBoostClassifier(iterations=?, learning_rate=?, depth=?, eval_metric ='AUC', silent=True)
# )
# ```

# In[ ]:


tree = ExtraTreesClassifier(n_estimators=50, random_state=42)
tree.fit(X_train, y_train)
selected_features = np.argsort(tree.feature_importances_)[-300:].astype(str)


# In[ ]:


final_pipeline = make_pipeline(
    StandardScaler(),
    PCA(),
    PolynomialFeatures(degree=3),
    CatBoostClassifier(iterations=1000, eval_metric ='AUC', silent=True)
)

parameters = {
    'pca__n_components': [3, 5, 10],
    'catboostclassifier__learning_rate':  np.logspace(-3, -1, num=5),
    'catboostclassifier__depth'       :  [4, 6, 8]
}

final_model = GridSearchCV(estimator=final_pipeline, 
                        param_grid=parameters,
                        cv=3, 
                        scoring='roc_auc',
                        verbose=10)


# In[ ]:


final_model.fit(X_train[selected_features], y_train)


# In[ ]:


paint_res(final_pipeline, X_train[selected_features], X_val[selected_features], y_train, y_val)


# In[ ]:


final_model.best_params_


# In[24]:


pipe = make_pipeline(
    StandardScaler(),
    PCA(n_components=3),
    PolynomialFeatures(3),
    CatBoostClassifier(iterations=5500, learning_rate=0.009, l2_leaf_reg=1, depth=7, eval_metric ='AUC')
)


# In[27]:


pipe.fit(X_train, y_train)


# In[29]:


paint_res(pipe, X_train, X_val, y_train, y_val)


# ## Дообучение и печать результатов

# In[30]:


result_pipeline = make_pipeline(
    StandardScaler(),
    PCA(n_components=3),
    PolynomialFeatures(3),
    CatBoostClassifier(iterations=5500, learning_rate=0.009, l2_leaf_reg=1, depth=7, eval_metric ='AUC')
)


# In[31]:


result_pipeline.fit(data.drop(columns = ['target']), data['target'])


# In[33]:


test = pd.read_csv('/kaggle/input/dataset-test/test.csv')


# In[46]:


predictions = result_pipeline.predict(test.drop(columns = ['id']))


# In[52]:


sumb = pd.DataFrame()
sumb['id'] = test['id']
sumb['target'] = predictions
sumb.to_csv('submission.csv')

