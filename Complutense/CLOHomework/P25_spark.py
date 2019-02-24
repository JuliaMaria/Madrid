from pyspark import SparkConf, SparkContext
import string
from pyspark.sql import SQLContext

conf = SparkConf().setMaster('local').setAppName('Meteorite')
sc = SparkContext(conf = conf)
sqlContext = SQLContext(sc)

df = sqlContext.read.csv("Meteorite_Landings.csv", header=True).select('recclass', 'mass (g)').selectExpr("recclass as recclass", "`mass (g)` as mass")
df = df.filter(df.mass.isNotNull()).rdd
sumCount = df.combineByKey(lambda value: (value, 1),
                             lambda x, value: (float(x[0]) + float(value), x[1] + 1),
                             lambda x, y: (x[0] + y[0], x[1] + y[1]))
averageByKey = sumCount.map(lambda (label, (value_sum, count)): (label, float(value_sum) / float(count))).sortByKey()
averageByKey.saveAsTextFile('out5.txt')


