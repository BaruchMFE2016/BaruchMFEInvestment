#!/usr/bin

./bcf.py -u r3000 \
         -p price \
         -m 6-52 \
         -f beta_adj \
         -f vol10 \
         -f market_cap \
         -d industry \
         -v 300000 \
         -z gauss \
         -r 20160101-20160101
