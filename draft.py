import sys
import itertools
import logging
import pandas as pd
import numpy as np
from math import sqrt
from operator import add
from os.path import join, isfile, dirname

from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession

from pyspark.mllib.recommendation import ALS
from pyspark.mllib.fpm import FPGrowth
from textblob import TextBlob
from textblob.classifiers import NaiveBayesClassifier
import json


spark = SparkSession \
    .builder \
    .appName("Python Spark SQL basic example") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

path_business ="/Users/yizhan/Desktop/CS181/yelp_dataset/sampleb.json"
df_business=spark.read.json(path_business)

df_filt_business = df_business.select(df_business["business_id"],df_business["categories"])
sample = df_filt_business.rdd.collect()


path_review="/Users/yizhan/Desktop/CS181/yelp_dataset/review.json"
df_review=spark.read.json(path_review)

df_filt_review=df_review.select(df_review["business_id"], df_review["text"])
totalsample=df_filt_business.join(df_filt_review, df_filt_business.business_id==df_filt_review.business_id).select(df_filt_business.business_id, df_filt_business.categories, df_filt_review.text)
totalsample.printSchema()




with open("/Users/yizhan/Desktop/newdata.json", "w") as f:
	totalSampleList = totalsample.toJSON().collect()
	for sample in totalSampleList:
		f.write(sample + "\n")

# # joinRDD=df_filt_business.join(df_filt_review)
# # df_filt=joinRDD.select(joinRDD["categories"], joinRDD["text"])
# # df_filt.printSchema()
# df_review_raw = spark.read.json(path_review)
# catDF = df_review_raw.select(df_review_raw["text"])
# catDF_iter = catDF.rdd.collect()
# for item in catDF.rdd.first().text:
    # print (item)

# from pyspark.mllib.feature import HashingTF, IDF

# # Load documents (one per line).
# documents = sc.textFile("data/mllib/kmeans_data.txt").map(lambda line: line.split(" "))

# hashingTF = HashingTF()
# tf = hashingTF.transform(documents)

# # While applying HashingTF only needs a single pass to the data, applying IDF needs two passes:
# # First to compute the IDF vector and second to scale the term frequencies by IDF.
# tf.cache()
# idf = IDF().fit(tf)
# tfidf = idf.transform(tf)

# # spark.mllib's IDF implementation provides an option for ignoring terms
# # which occur in less than a minimum number of documents.
# # In such cases, the IDF for these terms is set to 0.
# # This feature can be used by passing the minDocFreq value to the IDF constructor.
# idfIgnore = IDF(minDocFreq=2).fit(tf)
# tfidfIgnore = idfIgnore.transform(tf)




