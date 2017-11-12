import sys
import itertools
import logging
from math import sqrt
from operator import add
from os.path import join, isfile, dirname

from pyspark import SparkConf, SparkContext
from pyspark.mllib.recommendation import ALS
from pyspark.sql import SparkSession

import json

from pycorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP('http://localhost:9000')

spark = SparkSession \
    .builder \
    .appName("Python Spark SQL basic example") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

sc=spark.SparkContext(conf=conf)
path_business ="yizhan/Desktop/CS181/yelp dataset/buiness.json"
df_business=spark.read.json(path_business)
print(df_business.first())
df_filt_business = df.select(df["business_id"], df["name"], df["categories"])


path_review="yizhan/Desktop/CS181/yelp dataset/review.json"
df_review=spark.read.json(path_review)

df.filt_review=df.select(df["review_id"], df["user_id"], df["business_id"], df["text"])




