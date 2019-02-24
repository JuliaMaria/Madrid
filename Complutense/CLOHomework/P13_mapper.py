#!/usr/bin/python

import sys
import re

for line in sys.stdin:
    data = re.split(r",", line)
    if re.search('[a-zA-Z]', data[0]) is None and re.search('[a-zA-Z]', data[4]) is None:
        print( data[0] + "\t" + data[4] )


