#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install xgboost')


# In[2]:


import warnings
warnings.filterwarnings('ignore')

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier


# In[3]:


df = pd.read_csv('data_processed.csv')
df.head()


# # Criando o label

# In[4]:


df_received = df[df['event'] == 'offer received'].copy()


# In[5]:


df_completed = df[df['event'] == 'offer completed']


# In[6]:


df_received['offer_completed'] = df_received.apply(
    lambda row: 1 if ((df_completed['account_id'] == row['account_id']) & 
                      (df_completed['offer_id'] == row['offer_id'])).any() else 0,
    axis=1
)


# # Filtro de ofertas que não sejam de informação

# In[7]:


df_model = df_received[df_received['offer_type'] != 'informational'].copy()


# # Codificação e seleção de features

# In[8]:


df_model = pd.get_dummies(df_model, columns=['gender'], prefix='gender')
df_model = pd.get_dummies(df_model, columns=['offer_type'], prefix='offer_type')


# In[9]:


features = [
    'age', 'credit_card_limit', 'reward', 'discount_value', 'duration',
    'min_value', 'web', 'email', 'mobile', 'social',
    'gender_F', 'gender_M', 'gender_O', 'offer_type_bogo', 'offer_type_discount'
]


# In[10]:


X = df_model[features]
y = df_model['offer_completed']


# # Split de dados em treino e teste

# In[11]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# # Modelo

# In[12]:


model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


# # Métricas/Avaliação

# In[13]:


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Acurácia: {accuracy:.4f}")
print(f"Precisão: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")


# Interpretabilidade das métricas: 
# - Precision alta: menos envio de cupons "inúteis" → economia com descontos.
# - Recall alto: mais clientes que converteriam sendo atingidos → mais receita.
# 
# Ou seja:
# O modelo (F1 = 0.78) identifica 78% dos casos positivos de forma eficiente. 
# O que pode reduzir desperdício de envio de ofertas
# & aumentar conversões, assim impactando diretamente a receita.

# # Importâncida das features / Interpretabilidade

# In[14]:


import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)

shap.summary_plot(shap_values, X_train)


# Features:
# - credit_card_limit
#     - Impacto mais significativo no modelo.
#     - Valores altos de limite (pontos vermelhos) estão associados a SHAP positivos: clientes com maior limite tendem a completar mais ofertas.
#      - Valores baixos (pontos azuis) podem reduzir a probabilidade de conversão.
# 
# - age
#     - SHAP positivo: clientes mais velhos têm maior propensão a completar ofertas.
#     - Distribuição concentrada à direita sugere que idade é um fator consistente para conversão.
# 
# - discount_value
#     - Descontos maiores estão correlacionados com maior taxa de conversão (como esperado).
# 
# - canais de marketing
#     - social e web: impacto positivo moderado. Ofertas veiculadas por redes sociais e web são mais efetivas.
#     - mobile e email: impacto próximo de zero. Sugere que esses canais são menos eficazes para conversão.
# 
# - gênero
#      - homens respondem melhor às ofertas que mulheres ou nulos.
# 

# Insights:
# - Priorização de clientes com alto limite de cartão e idade mais avançada.
# - Reduza o min_value exigido para ativar ofertas.
# - Usar canais sociais e web para veiculação.
