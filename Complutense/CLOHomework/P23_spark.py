from pyspark import SparkConf, SparkContext
import string
from pyspark.sql import SQLContext

conf = SparkConf().setMaster('local').setAppName('Stock')
sc = SparkContext(conf = conf)
sqlContext = SQLContext(sc)

df = sqlContext.read.csv("GOOG.csv", header=True).select('Date', 'Close').rdd.map(lambda x: (x[0].split('-')[0], x[1])).filter(lambda x: int(x[0]) >= 2009)
sumCount = df.combineByKey(lambda value: (value, 1),
                             lambda x, value: (float(x[0]) + float(value), x[1] + 1),
                             lambda x, y: (x[0] + y[0], x[1] + y[1]))
averageByKey = sumCount.map(lambda (label, (value_sum, count)): (label, float(value_sum) / float(count))).sortByKey()
averageByKey.saveAsTextFile('out3.txt')


