import sys
import itertools
import logging
import pandas as pd
import random
from math import sqrt
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

datapath="/Users/yizhan/Desktop/newdata.json"
data1=spark.read.json(datapath)

total=[]
for item in data1.rdd.collect():
	vk=[]
	text=item['text']
	vk.append(text)
	# print(vk)
	cat=item['categories']
	# print(type(cat))
	# for i in cat:
	# 	clist=TextBlob(i)
	mergelist=vk+cat
	mergelist=tuple(mergelist)
	total.append(mergelist)

# 	# string=TextBlob(item)
# 	# wlist=string.words
# 	# print(wlist)
dwf=total[3]
defz=list(dwf[1:])

tt=TextBlob(dwf[0].replace("'",""))
q=[]
p=[]
for a in defz:
	b=TextBlob(a)
	c=b.words
	for k in c:
		u=k.synsets
		q.append(u)
q=sum(q, [])
for z in tt.words:
	y=z.synsets
	p.append(y)
print(q, p)
s=0
v=0
x=0
for h in p:
	for i in h:
		for j in q:
			z=i.path_similarity(j)
			
			if(z is None):
				z=0
			else:
				v+=1
		s+=z
	
		ind=h.index(i)
		wo=	tt.words[ind]
	x+=s
	average_simi=x/v
	print(average_simi, wo, j)

# print(q)

# final=[]
# for line in total:
# 	temp=TextBlob(line[0].replace("'", ""))
# 	cate=list(line[1:])
# 	l=len(temp)
# 	t=[]
# 	for w in temp.words:
# 		k=w.synsets
# 		t.append(k)
# 	cc=[]
# 	for c in cate:
# 		b=TextBlob(c)
# 		d=b.words
# 		for k in d:
# 			u=k.synsets
# 			cc.append(u)
# 	cc=sum(cc, [])

# 	for m in t:
# 		for f in m:
# 			for n in cc:
# 				z=f.path_similarity(n)
# 				if z>0.5:
# 					p=tuple(f, n)
# 					final.append(p)
# print(final)

# traindata=total[:100]

# cl=NaiveBayesClassifier(traindata)
# k=cl.classify(total[300][0])
# print(k)
# test=total[:100]
# test1=test[1][1]
# print(test1)
# print(data1.first().text)
# with open('/Users/yizhan/Desktop/newdata.json', 'r') as f:
# 	cl=NaiveBayesClassifier(f.rdd.text , format="json")


# print(blob)