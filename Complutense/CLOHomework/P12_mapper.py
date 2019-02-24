#!/usr/bin/python

import sys
import re

for line in sys.stdin:
    result = re.search('"(.*)"', line)
    result2 = re.search(' (.*) ', result.group(1))
    if result2 is not None:
        print( result2.group(1) + "\t1" )

