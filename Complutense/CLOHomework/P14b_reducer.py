#!/usr/bin/python

import sys
import math

previous_rating = 1
movie_id = []

for line in sys.stdin:
    rating, id = line.split('\t')

    if float(rating) > float(previous_rating):
        print str(previous_rating) + '\t' + str(movie_id)
        previous_rating = previous_rating + 1
        movie_id = [id.rstrip()]

    if not id.rstrip() in movie_id:
        movie_id.append(id.rstrip())
        
print str(previous_rating) + '\t' + str(movie_id)
