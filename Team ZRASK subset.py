#!/usr/bin/env python
# coding: utf-8

# In[1]:


#We are creating a mount while will allow us to access our dataset through our Amazon S3 account

import urllib
ACCESS_KEY = "******"
SECRET_KEY = "******"
ENCODED_SECRET_KEY = SECRET_KEY.replace("/", "%2F")    
AWS_BUCKET_NAME = "*****"
MOUNT_NAME = "******"

#Remove the below line's "#" if this is the first time running this Notebook, after that you will need to comment it out (Use a #), as there will already be a mount created
dbutils.fs.mount("s3a://%s:%s@%s" % (ACCESS_KEY, ENCODED_SECRET_KEY, AWS_BUCKET_NAME), "/mnt/%s" % MOUNT_NAME)

display(dbutils.fs.ls("/mnt/%s" % MOUNT_NAME))


# #### Read the User dataset from JSON format and storing it as Pyspark dataframe

# In[3]:


from pyspark.sql.types import StructType, StructField, IntegerType, StringType, BooleanType, FloatType, ArrayType, TimestampType


# In[4]:


#Reading the Amazon S3 User dataset from JSON format and storing it as Pyspark dataframe
user_schema = StructType([StructField('average_stars', FloatType(), True),
                         StructField('compliment_cool', IntegerType(), True),
                         StructField('compliment_funny', IntegerType(), True),
                         StructField('compliment_hot', IntegerType(), True),
                         StructField('compliment_list', IntegerType(), True),
                         StructField('compliment_more', IntegerType(), True),
                         StructField('compliment_note', IntegerType(), True),
                         StructField('compliment_photos', IntegerType(), True),
                         StructField('compliment_plain', IntegerType(), True),
                         StructField('compliment_profile', IntegerType(), True),
                         StructField('compliment_writer', IntegerType(), True),
                         StructField('cool', IntegerType(), True),
                         StructField('elite', StringType(), True),
                         StructField('fans', IntegerType(), True),
                         StructField('friends', StringType(), True),
                         StructField('funny', IntegerType(), True),
                         StructField('name', StringType(), True),
                         StructField('review_count', IntegerType(), True),
                         StructField('useful', IntegerType(), True),
                         StructField('user_id', StringType(), True),
                         StructField('yelping_since', TimestampType(), True)])

Users_df = spark.read.json("dbfs:/mnt/inst767-mount/user.json", schema = user_schema)

Users_df = Users_df.withColumnRenamed('review_count', 'user_review_count')

#Double check the schema of the dataframe represnting the Users dataset
Users_df.printSchema()

#Keeping only the columns that we want to use for our analysis
Users_refined_df = Users_df.select([c for c in Users_df.columns if c in ['user_id', 'average_stars', 'fans', 'user_review_count']])

display(Users_refined_df)


# #### Reading the Amazon S3 Checkins dataset from JSON format and storing it as Pyspark dataframe

# In[6]:


#Reading the Amazon S3 Checkins dataset from JSON format and storing it as Pyspark dataframe

Checkin_df = spark.read.json("dbfs:/mnt/inst767-mount/checkin.json")

display(Checkin_df)


# #### Reading the Amazon S3 Reviews dataset from JSON format and storing it as Pyspark dataframe

# In[8]:


# #Reading the Amazon S3 Reviews dataset from JSON format and storing it as Pyspark dataframe

# reviews_schema = StructType([StructField('business_id', StringType(), True),
#                             StructField('cool', IntegerType(), True),
#                             StructField('date', TimestampType(), True),
#                             StructField('funny', IntegerType(), True),
#                             StructField('review_id', StringType(), True),
#                             StructField('stars', IntegerType(), True),
#                             StructField('text', StringType(), True),
#                             StructField('useful', IntegerType(), True),
#                             StructField('user_id', StringType(), True)])

reviews_df = spark.read.json("dbfs:/mnt/inst767-mount/review.json")

reviews_df = reviews_df.withColumn('cool_integer', reviews_df['cool'].cast(IntegerType())).drop('cool')
reviews_df = reviews_df.withColumn('funny_integer', reviews_df['funny'].cast(IntegerType())).drop('funny')
reviews_df = reviews_df.withColumn('stars_integer', reviews_df['stars'].cast(IntegerType())).drop('stars')
reviews_df = reviews_df.withColumn('useful_integer', reviews_df['useful'].cast(IntegerType())).drop('useful')

reviews_df = reviews_df.withColumnRenamed('cool_integer', 'cool')
reviews_df = reviews_df.withColumnRenamed('funny_integer', 'funny')
reviews_df = reviews_df.withColumnRenamed('stars_integer', 'stars')
reviews_df = reviews_df.withColumnRenamed('useful_integer', 'useful')

display(reviews_df)


# In[9]:


@udf(returnType=IntegerType())
def word_count(col):
  return len(col)

reviews_df = reviews_df.withColumn('review_len', word_count('text'))


# In[10]:


reviews_df.count()


# #### Reading the Amazon S3 Tips dataset from JSON format and storing it as Pyspark dataframe

# In[12]:


#Reading the Amazon S3 Tips dataset from JSON format and storing it as Pyspark dataframe

tips_schema = StructType([StructField('business_id', StringType(), True),
                         StructField('compliment_count', IntegerType(), True),
                         StructField('date', TimestampType(), True),
                         StructField('text', StringType(), True),
                         StructField('user_id', StringType(), True)])

tips_df = spark.read.json("dbfs:/mnt/inst767-mount/tip.json")

display(tips_df)


# #### Reading the Amazon S3 Business dataset from JSON format and storing it as Pyspark dataframe

# In[14]:


#Reading the Amazon S3 Business dataset from JSON format and storing it as Pyspark dataframe

business_schema = StructType([StructField('address', StringType(), True),
                             StructField('attributes', StructType(), True),
                             StructField('business_id', StringType(), True),
                             StructField('categories', StringType(), True),
                             StructField('city', StringType(), True),
                             StructField('hours', StructType(), True),
                             StructField('is_open', IntegerType(), True),
                             StructField('latitude', FloatType(), True),
                             StructField('longitude', FloatType(), True),
                             StructField('name', StringType(), True),
                             StructField('postal_code', StringType(), True),
                             StructField('review_count', IntegerType(), True),
                             StructField('stars', FloatType(), True),
                             StructField('state', StringType(), True)])

business_df = spark.read.json("dbfs:/mnt/inst767-mount/business.json", schema = business_schema)

business_df = business_df.withColumnRenamed('review_count', 'business_review_count')

display(business_df)


# ### Preprocessing the business dataframe

# In[16]:


#UDF that counts # of rows and # of columns
def shape(df):
  return [df.count(), len(df.columns)]

#Drop the attributes and hours columns, then check new schema
business_df = business_df.drop('attributes')
business_df = business_df.drop('hours')

#Checking the # of null/missing values in the Business dataset
from pyspark.sql.functions import isnan, when, count, col
business_df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in business_df.columns]).show()

# Drop rows with null values in categories column
business_df = business_df.dropna(subset = 'categories')
# business_df.show(5)

#Only keeping the businesses that are restaurants or are food related
from pyspark.sql.functions import udf
from pyspark.sql.types import BooleanType

@udf(returnType=BooleanType())
def filter_restaurants(col):
  for category in col.split(','):
    if 'restaurant' in category.lower() or 'food' in category.lower():
      return True
  return False

@udf(returnType=BooleanType())
def filter_non_restaurants(col):
  category = ' '.join(col.split(','))
  if 'restaurant' not in category.lower() and 'food' not in category.lower():
    return True
  return False

#Reviewing businesses that are restaurants and are food related
business_df_restaurants = business_df.filter(filter_restaurants('categories'))

#Reviewing businesses that are not restaurants and are not food related
business_df_nonRestaurants = business_df.filter(filter_non_restaurants('categories'))

#Determining the # of unique categories from our restaurants dataframe as well as our non-restaurants non-restaurants dataframe, in case we find other keywords that should be used
def get_unique(df):
  """
  Returns a list of unique categories from the df
  """
  unique_categories = {}
  
  for row in df.select('categories').rdd.collect():
    categories = row.asDict()['categories']
    for category in categories.split(','):
      try:
        unique_categories[category.lower().strip()] += 1
      except:
        unique_categories[category.lower().strip()] = 0
        
  return unique_categories
    
unique_category_nonRestaurants = get_unique(business_df_nonRestaurants)
unique_category_restaurants = get_unique(business_df_restaurants)

print('Number of unique categories in restaurants df are {}'.format(len(unique_category_restaurants)))  
print('Number of unique categories in non restaurants df are {}'.format(len(unique_category_nonRestaurants))) 


# In[17]:


display(business_df_restaurants.select('state').distinct())


# In[18]:


business_df_restaurants.createOrReplaceTempView('business_restaurants_VIEW')


# In[19]:


get_ipython().run_line_magic('sql', 'select state, count(business_id) from business_restaurants_VIEW group by state order by count(business_id) desc')


# ### We choose a subset of the data. Select the following states: AZ, OH

# In[21]:


filtered_business_df = business_df_restaurants.filter((business_df_restaurants.state == 'AZ') | (business_df_restaurants.state == 'OH'))

display(filtered_business_df)


# In[22]:


shape(filtered_business_df)


# ### Join the Review, Business and Users dataframes

# In[24]:


#Checking unique user_id count, making sure there are no duplicate records. They are they same, but the code is commented out so the cell runs faster
#Users_refined_df.select("user_id").distinct().count()
#Users_refined_df.select("user_id").count()

#Rename the stars column in both the review and business dataframes, so that there is no confusion when joining
reviews_df2 = reviews_df
reviews_df2 = reviews_df2.withColumnRenamed('stars', 'review_stars')
business_df_restaurants2 = filtered_business_df
business_df_restaurants2 = business_df_restaurants2.withColumnRenamed('stars', 'business_stars')

#Making a copy of the Users dataframe
#Users_refined_df2 = Users_refined_df

#Join the Review dataframe with the Business dataframe
#review_analysis_df = reviews_df2
review_analysis_df = reviews_df2.join(business_df_restaurants2, 'business_id', 'inner')

#Join the above review_analysis_df dataframe with the Users dataframe, so that we have one combined data frame that is per review
review_analysis_df = review_analysis_df.join(Users_refined_df, 'user_id', 'inner')



# In[25]:


display(review_analysis_df)


# In[26]:


shape(review_analysis_df)


# ### Vader Sentiment

# In[28]:


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyser = SentimentIntensityAnalyzer()

# positive sentiment - compound score >= 0.05
# neutral sentiment - compound score (-0.05, +0.05)
# negative sentiment - compund score <= -0.05


# User defined function to calculate sentiment score for each review

# In[30]:


from pyspark.sql.types import FloatType

@udf(returnType = FloatType())
def sentiment_score(col):
  scores = analyser.polarity_scores(col)
  return scores['compound']


review_analysis_df = review_analysis_df.withColumn('sentiment_score', sentiment_score('text'))


#review_analysis_df.select('text', sentiment_score('text').alias('sentiment_score'))['sentiment_score']

display(review_analysis_df)


# #### Create an accuracy column
# review_stars :<br>
#               1,2 - Negative<br>
#                 3 - Neutral<br>
#               4,5 - positive<br>
#               
# positive sentiment - compound score >= 0.05<br>
# neutral sentiment - compound score (-0.05, +0.05)<br>
# negative sentiment - compund score <= -0.05<br>

# In[32]:


@udf(returnType = IntegerType())
def accuracy(reviewStars, sentimentScore):
  if (reviewStars < 3 and sentimentScore <= -0.05) or (reviewStars == 3 and sentimentScore > -0.05 and sentimentScore < 0.05) or (reviewStars > 3 and sentimentScore >= 0.05):
    return 1
  else:
    return 0
  
review_analysis_df = review_analysis_df.withColumn('accurate', accuracy('review_stars', 'sentiment_score'))

display(review_analysis_df)


# In[33]:


review_analysis_df.rdd.getNumPartitions()


# In[34]:


review_analysis_df.repartition(30).createOrReplaceTempView('reviewAnalysisVIEW')


# In[35]:


spark.catalog.cacheTable('reviewAnalysisVIEW')


# In[36]:


# Call .count() to materialize the cache
spark.table("reviewAnalysisVIEW").count()


# In[37]:


reviewAnalysisDF = spark.table("reviewAnalysisVIEW")


# In[38]:


# Note that the full scan + count in memory takes < 1 second!

reviewAnalysisDF.count()


# In[39]:


spark.catalog.isCached("reviewAnalysisVIEW")


# In[40]:


display(reviewAnalysisDF.groupby('accurate').count())


# In[41]:


reviewAnalysisDF.write.parquet('/tmp/reviewAnalysisDf')


# In[42]:


display(reviewAnalysisDF)


# In[43]:


display(reviewAnalysisDF.select('review_stars').distinct())


# In[44]:


get_ipython().run_line_magic('fs', 'ls tmp/reviewAnalysisDf')


# ### 3: Vector Assembly
# Once we are through with the encoder creation step, it is time to essemble the encoders and all the input and output columns to form a final vector_generator that will be passed as input to the machine learning pipeline.

# In[46]:


import pyspark.ml.feature as ft

featuresCreator = ft.VectorAssembler(inputCols=['cool', 'funny', 'useful', 'is_open', 'business_review_count', 'business_stars', 'average_stars', 'fans', 'user_review_count', 'sentiment_score'],outputCol='features')


# ### 4: Estimator Creation
# This is the step where we select the machine learning model that we wish to utilize. Here, we create an Estimator object that contains the machine learning model along with all the hyper optimization parameters that need to be passed to it. Here, we are using LogisticRegression.

# In[48]:


import pyspark.ml.classification as cl


logistic_regression_model = cl.LogisticRegression(maxIter=10,regParam=0.01,labelCol='review_stars', family = 'multinomial')
print(type(logistic_regression_model))


# ### 5: Pipeline Creation

# In[50]:


from pyspark.ml import Pipeline


pipeline = Pipeline(stages=[featuresCreator,logistic_regression_model])
print(type(pipeline))


# ### 6: Dataset Splitting

# In[52]:


train, test = reviewAnalysisDF.randomSplit([0.7, 0.3], seed=666)


# ### 7: Model Fitting

# In[54]:


# This creates a model by training on the births_train dataset using the Pipeline we created
model = pipeline.fit(train)
print(type(model))

# This step uses the model that we created in the previous step to generate the output column for births_test dataset
test_model = model.transform(test)
print(type(test_model))


# In[55]:


display(test_model.select('prediction').distinct())


# We observe that our model does not give any 2 star ratings. Let's check how accurate it is.

# ### 8: Performance Evaluation

# In[58]:


import pyspark.ml.evaluation as ev


# Create an object of the Evaluator class that evaluates our model based on the rawPredictionCol and the label that we need to predict
evaluator = ev.MulticlassClassificationEvaluator(predictionCol='prediction',labelCol='review_stars')

# Extract the needed evaluation criteria from the object
print(evaluator.evaluate(test_model,{evaluator.metricName: 'accuracy'}))
# print(evaluator.evaluate(test_model,{evaluator.metricName: 'areaUnderPR'}))


# We observe an accuracy of 51 percent on this baseline model

# In[60]:


get_ipython().run_line_magic('fs', 'ls reviewAnalysisDf.tsv')


# In[61]:





# In[62]:




