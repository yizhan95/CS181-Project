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

# path_business ="/Users/yizhan/Desktop/CS181/yelp_dataset/sampleb.json"
# df_business=spark.read.json(path_business)

# df_filt_business = df_business.select(df_business["business_id"],df_business["categories"])
# sample = df_filt_business.rdd.collect()


path_review="/Users/yizhan/Desktop/CS181/yelp_dataset/star.json"
df_review=spark.read.json(path_review)
df_filt_review=df_review.select(df_review["text"], df_review["stars"])

# totalsample=df_filt_business.join(df_filt_review, df_filt_business.business_id==df_filt_review.business_id).select(df_filt_business.business_id, df_filt_business.categories, df_filt_review.text)





with open("/Users/yizhan/Desktop/newdatastar.json", "w") as f:
	totalSampleList = df_filt_review.toJSON().collect()
	for sample in totalSampleList:
		f.write(sample+"\n")





