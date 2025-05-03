# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook

# COMMAND ----------

from pyspark.sql import functions as F

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Ma­ni­pu­la­ção e lim­pe­za dos da­dos

# COMMAND ----------

path = 's3://dev-ifood-ml-sagemaker/ifood-ml-ds-groceries/analysis'

path_transactions = 'transactions.json'
path_profile = 'profile.json'
path_offers = 'offers.json'

# COMMAND ----------

# MAGIC %md
# MAGIC ### Dados de transações

# COMMAND ----------

df_transactions = spark.read.json(f'{path}/{path_transactions}')
display(df_transactions.limit(5))

# COMMAND ----------

display(df_transactions.filter(F.col('value.offer_id').isNotNull()).filter(F.col('value.offer id').isNotNull()))

# COMMAND ----------

df_transactions = df_transactions \
  .withColumn('ammount', F.col('value.amount')) \
  .withColumn('offer_id',
    F.when(F.col('value.offer_id').isNotNull() & (F.col('value.offer_id') != ""), F.col('value.offer_id'))
    .otherwise(F.col('value.offer id'))) \
  .withColumn('reward', F.col('value.reward')) \
  .drop('value') 

display(df_transactions)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Dados de perfil dos usuários

# COMMAND ----------

df_profile = spark.read.json(f'{path}/{path_profile}')
display(df_profile.limit(5))

# COMMAND ----------

df_profile = df_profile \
  .withColumn('registered_on', F.to_date('registered_on', 'yyyyMMdd')) \
  .withColumnRenamed('id', 'account_id')

display(df_profile)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Dados de ofertas

# COMMAND ----------

df_offers = spark.read.json(f'{path}/{path_offers}')
display(df_offers)

# COMMAND ----------

df_offers = df_offers \
  .withColumnRenamed('id', 'offer_id') \
  .withColumn("web", F.array_contains(F.col("channels"), "web").cast("int")) \
  .withColumn("email", F.array_contains(F.col("channels"), "email").cast("int")) \
  .withColumn("mobile", F.array_contains(F.col("channels"), "mobile").cast("int")) \
  .withColumn("social", F.array_contains(F.col("channels"), "social").cast("int")) \
  .drop('channels')

display(df_offers)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Checagem de linhas para unificação

# COMMAND ----------

print(f"O df_transactions tem {df_transactions.count()} linhas.")
print(f"O df_profile tem {df_profile.count()} linhas.")
print(f"O df_offers tem {df_offers.count()} linhas.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Unificação do da­ta­set

# COMMAND ----------

df = df_transactions \
  .join(df_profile, on='account_id', how='inner') \
  .join(df_offers, on='offer_id', how='inner')

# COMMAND ----------

print(f"O dataset unificado tem {df.count()} linhas.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Por que perdemos linhas na unificação dos dados?

# COMMAND ----------

df_teste_1 = df_transactions.join(df_profile, on='account_id', how='inner')
display(df_teste_1.count())

# COMMAND ----------

# MAGIC %md
# MAGIC Certo, todas as transações possuem um perfil de usuário. 

# COMMAND ----------

df_teste_2 = df_transactions.join(df_offers, on='offer_id', how='inner')
display(df_teste_2.count())

# COMMAND ----------

# MAGIC %md 
# MAGIC Nem toda transação possui uma oferta, o que justifica a perda de linhas na unificação dos dados. Isso significa que nem todas as transações possuem uma oferta, o que faz sentido. 
# MAGIC Então a unificação do dataset será refeito para garantir que todas as transações sejam salvas no dataset final.

# COMMAND ----------

df = df_transactions \
  .join(df_profile, on='account_id', how='left') \
  .join(df_offers, on='offer_id', how='left')

display(df)

# COMMAND ----------

print(f"O dataset unificado tem {df.count()} linhas.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Entendendo os eventos

# COMMAND ----------

display(df.select('event').dropDuplicates())

# COMMAND ----------

display(df.filter(F.col('event') == 'transaction'))

# COMMAND ----------

display(df.filter(F.col('event') == 'transaction').select('offer_id').dropDuplicates())

# COMMAND ----------

# MAGIC %md
# MAGIC Quando o evento é 'transaction' não temos nenhuma oferta vinculada.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Oferta recebida

# COMMAND ----------

display(df.filter(F.col('event') == 'offer received'))

# COMMAND ----------

display(df.filter(F.col('event') == 'offer received').filter(F.col('offer_id').isNull()))

# COMMAND ----------

display(df.filter(F.col('event') == 'offer received') \
  .groupBy('offer_type').count() \
  .withColumn('pct_type_by_offer_received', F.round((F.col('count') / df.filter(F.col('event') == 'offer received').count()) * 100, 4)))

# COMMAND ----------

display(df.filter(F.col('event') == 'offer received').select('ammount').dropDuplicates())

# COMMAND ----------

display(df.filter(F.col('event') == 'offer received').select('reward').dropDuplicates())

# COMMAND ----------

# MAGIC %md
# MAGIC Quando a oferta foi recebida, apenas, sabemos o tipo da oferta, mas não temos, até então, a compra realizada (logo não temos valor da transição e do desconto.)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Oferta visualizada

# COMMAND ----------

display(df.filter(F.col('event') == 'offer viewed'))

# COMMAND ----------

display(df.filter(F.col('event') == 'offer viewed') \
  .groupBy('offer_type').count() \
  .withColumn('pct_type_by_offer_viewed', F.round((F.col('count') / df.filter(F.col('event') == 'offer viewed').count()) * 100, 4)))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Oferta completada

# COMMAND ----------

display(df.filter(F.col('event') == 'offer completed'))

# COMMAND ----------

display(df.filter(F.col('event') == 'offer completed') \
  .groupBy('offer_type').count() \
  .withColumn('pct_type_by_offer_completed', F.round((F.col('count') / df.filter(F.col('event') == 'offer completed').count()) * 100, 4)))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Verificação

# COMMAND ----------

display(df.filter(F.col('offer_id').isNotNull()).groupBy('offer_id', 'account_id').count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Salvando os dados

# COMMAND ----------

df.write.mode('overwrite').option('header', 'true').csv(f'{path}/processed_df')