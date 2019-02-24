#!/usr/bin/python

import sys
import re

for line in sys.stdin:
    arg = sys.argv[1]
    line = re.sub( r'^\W+|\W+$', '', line )
    if arg in line:
        print("" + line)


