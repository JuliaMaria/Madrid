#!/usr/bin/python

import sys
import re

for line in sys.stdin:
    data = re.split(r",", line)
    if re.search('[a-zA-Z]', data[1]) is None and re.search('[a-zA-Z]', data[2]) is None:
        print( data[1] + "\t" + data[2] )


