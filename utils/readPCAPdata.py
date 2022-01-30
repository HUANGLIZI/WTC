import csv
from scapy.all import *

pkts = rdpcap("20201214.pcap")
pkt0 =pkts[0]

headers=['dst','src','type','version','tc','f1','plen','nh','hlim','IPsrc','IPdst','sport','dport','seq','ack',
         'dataofs','reserved','flags','window','chksum','urgptr','optionsmss','optionsNOP0','optionsWScale',
         'optionsNOP1','optionsNOP2','optionsSAckOK']

a1=pkt0['Ethernet'].dst
a2=pkt0['Ethernet'].src
a3=pkt0['Ethernet'].type

b1=pkt0['IPv6'].version
b2=pkt0['IPv6'].tc
b3=pkt0['IPv6'].fl
b4=pkt0['IPv6'].plen
b5=pkt0['IPv6'].nh
b6=pkt0['IPv6'].hlim
b7=pkt0['IPv6'].src
b8=pkt0['IPv6'].dst
c1=pkt0['TCP'].sport
c2=pkt0['TCP'].dport
c3=pkt0['TCP'].seq
c4=pkt0['TCP'].ack
c5=pkt0['TCP'].dataofs
c6=pkt0['TCP'].reserved
c7=pkt0['TCP'].flags
c8=pkt0['TCP'].window
c9=pkt0['TCP'].chksum
c10=pkt0['TCP'].urgptr
c11=pkt0['TCP'].options[0][1]
c12=pkt0['TCP'].options[1][1]
c13=pkt0['TCP'].options[2][1]
c14=pkt0['TCP'].options[3][1]
c15=pkt0['TCP'].options[4][1]
c16=pkt0['TCP'].options[5][1]

rows=[a1,a2,a3,b1,b2,b3,b4,b5,b6,b7,b8,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,c16]

with open('normal.csv','w',newline ='') as f:
    fcsv= csv.writer(f)
    fcsv.writerow(headers)
    fcsv.writerow(rows)
