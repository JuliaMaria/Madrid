from pyspark import SparkConf, SparkContext
import string
from pyspark.sql import SQLContext
from types import *
import math

conf = SparkConf().setMaster('local').setAppName('Movie')
sc = SparkContext(conf = conf)
sqlContext = SQLContext(sc)

df = sqlContext.read.csv("ratings.csv", header=True).select('movieId', 'rating').rdd.map(lambda row: tuple([x for x in row]))
sumCount = df.combineByKey(lambda value: (value, 1),
                             lambda x, value: (float(x[0]) + float(value), x[1] + 1),
                             lambda x, y: (x[0] + y[0], x[1] + y[1]))
averageByKey = sumCount.map(lambda (label, (value_sum, count)): (label, float(value_sum) / float(count)))
averageByKey = averageByKey.map(lambda (id, avg): (avg, id)).map(lambda (avg, id): (math.ceil(avg), id)).groupByKey().map(lambda x: (x[0], list(x[1]))).sortByKey()
averageByKey.saveAsTextFile('out4.txt')


