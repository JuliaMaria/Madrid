#!/usr/bin/python

import sys

previous = None
sum = 0
number = 0

for line in sys.stdin:
    key, value = line.split('\t')
    year = key.split('-')[0]

    if int(year) < 2009:
        continue
    if year != previous:
        if previous is not None:
            print previous + '\t' + str( round(sum/number, 2) )
        previous = year
        sum = 0
        number = 0

    sum = sum + float( value )
    number = number + 1

print previous + '\t' + str( round(sum/number, 2) )
