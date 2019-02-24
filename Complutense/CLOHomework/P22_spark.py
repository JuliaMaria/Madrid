from pyspark import SparkConf, SparkContext
import string
import re

conf = SparkConf().setMaster('local').setAppName('Log')
sc = SparkContext(conf = conf)

result = sc.textFile('access_log').map(lambda x: x.split(" ")[6])
result = result.map(lambda line: (line, 1))

count = result.reduceByKey(lambda a, b: a + b).sortByKey()
count.saveAsTextFile('out2.txt')
