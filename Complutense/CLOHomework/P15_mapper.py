#!/usr/bin/python

import sys
import re

for line in sys.stdin:
    data = re.split(r',(?!\s)', line)
    if re.search('[a-zA-Z]', data[4]) is None and not data[4] == "":
        print( data[3] + "\t" + data[4] )



