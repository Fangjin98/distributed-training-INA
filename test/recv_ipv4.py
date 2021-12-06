from scapy.all import *


iface = "eno6"

if __name__=="__main__":
    print("Sniffing on ", iface)
    print("Press Ctrl-C to stop...")
    sniff(iface=iface, prn=lambda p: p.show())
