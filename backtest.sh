#!bin/bash/
START_DATE='2014-01-01'
END_DATE='2016-12-31'
DATA_DIR='/home/derek-qi/Documents/R3000_Data/data/r3000/'

python backtest_main.py -s $START_DATE -e $END_DATE -d $DATA_DIR | tee ./log/backtest_log.txt