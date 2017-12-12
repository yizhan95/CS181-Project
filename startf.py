import sys
import itertools
import time
import logging
import pandas as pd
import random
import math
from operator import add
from os.path import join, isfile, dirname

from textblob import TextBlob
from textblob.classifiers import NaiveBayesClassifier
from textblob import Word
from textblob.wordnet import VERB
from textblob.wordnet import NOUN
from textblob.wordnet import Synset

import json
from pprint import pprint
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.mllib.recommendation import ALS
from pyspark.mllib.fpm import FPGrowth

spark = SparkSession \
    .builder \
    .appName("Python Spark SQL basic example") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

datapath="/Users/yizhan/Desktop/newdatastar.json"
data1=spark.read.json(datapath)
training, testing = data1.rdd.randomSplit([0.6, 0.4])


test1=[]
train1=[]
for item in training.collect():
	vk=[]
	text=item['text']
	vk.append(text)
	# print(vk)
	cat=item['stars']
	cat1=[int(cat)]
	# print(type(cat))
	# for i in cat:
	# 	clist=TextBlob(i)
	mergelist=vk+cat1
	mergelist[0].strip()
	mergelist=tuple(mergelist)
	train1.append(mergelist)

for item in testing.collect():
	vk=[]
	text=item['text']
	vk.append(text)
	# print(vk)
	cat=item['stars']
	cat1=[int(cat)]
	# print(type(cat))
	# for i in cat:
	# 	clist=TextBlob(i)
	mergelist=vk+cat1
	mergelist[0].strip()
	mergelist=tuple(mergelist)
	test1.append(mergelist)

cl=NaiveBayesClassifier(train1)

# # # result=cl.prob_classify(test)
# # # re=cl.classify(test)
# # # print(result.max(), re, score)
result=cl.accuracy(test1)
print(result)
# length=len(testing)
# case=[]
# corr=0
# for t in testing:
# 	test=t[0]
# 	score=t[1]
# 	result=cl.classify(test)
# 	prob_result=cl.prob_classify(test)
# 	if result==score or prob_result.max()==score:
# 		corr+=1
# 	acc=corr/length
# 	case.append(acc)
# print(case)


