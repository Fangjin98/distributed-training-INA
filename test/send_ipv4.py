#!/usr/bin/python

import os
import sys

from scapy.all import *
from scapy.layers.l2 import Ether
from scapy.layers.inet import IP
from scapy.layers.inet import UDP

try:
    ip_dst = sys.argv[1]
except:
    ip_dst = "172.16.170.1"

try:
    count = int(sys.argv[2], base=0)
except:
    count = 100

print("Sending %d IP packet(s) to %s" % (count, ip_dst))
p = (Ether(dst="00:11:22:33:44:55", src="48:df:37:5c:ff:b8") /
     IP(src="172.16.170.2", dst=ip_dst))

for i in range(count):
    sendp(p, iface="eno6", count=count)
