import csv
from scapy.all import *

pkts = rdpcap("zy-20201214.pcap")
headers=['dst','src','type','version','ihl','tos','len','id','flags','ttl','chksum','IPsrc','IPdst','sport','dport','seq','ack',
         'dataofs','reserved','flags','window','chksum','urgptr','optionsmss','optionsNOP0','optionsWScale']
with open('attack.csv','a',newline ='') as f:
    fcsv= csv.writer(f)
    fcsv.writerow(headers)
    for i in range(len(pkts)):
        pkt0 =pkts[i]
        if pkt0.payload.payload.name !='TCP':
            continue



        a1=pkt0['Ethernet'].dst
        a2=pkt0['Ethernet'].src
        a3=pkt0['Ethernet'].type

        b1=pkt0['IP'].version
        b2=pkt0['IP'].ihl
        b3=pkt0['IP'].tos
        b4=pkt0['IP'].len
        b5=pkt0['IP'].id
        b6=pkt0['IP'].flags
        b7=pkt0['IP'].ttl
        b8=pkt0['IP'].chksum
        b9=pkt0['IP'].src
        b10=pkt0['IP'].dst
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
        #c11=pkt0['TCP'].options[0][1]
        #c12=pkt0['TCP'].options[1][1]
        #c13=pkt0['TCP'].options[2][1]

        rows=[a1,a2,a3,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10]
        fcsv.writerow(rows)