from pyspark import SparkConf, SparkContext
import string
import sys

conf = SparkConf().setMaster('local').setAppName('DistGrep')
sc = SparkContext(conf = conf)
arg = sys.argv[1]
grep = sc.textFile('2701-0txt').map(lambda line: line).filter(lambda line: arg in line)
grep.saveAsTextFile('out1.txt')

