#!bin/bash/
START_DATE='2014-01-01'
END_DATE='2016-12-31'

python backtest_main.py -s $START_DATE -e $END_DATE | tee ./log/backtest_log.txt