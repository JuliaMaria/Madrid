#!/usr/bin/python

import sys
import re

for line in sys.stdin:
    id, rating = line.split( '\t' )
    print( rating.rstrip() + '\t' + id )


