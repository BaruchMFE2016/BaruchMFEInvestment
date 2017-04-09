# BaruchMFEInvestment
Baruch Investment Club project repository

## Dependencies
Python 3.5 or above with the following packages
- numpy
- pandas
- cvxopt
- yahoo_finance

## Project Description and Structures(planning)
This project aims at implementing a commonly used factor model applied to the equity market, introduce alpha mining framework, and could possibly serve as a research platform for future alpha research.

### Project framework
./backtest_main.py: main script for backtest
./setup/: contains script for environment setup, first step data cleaning etc.
./factor_mining/: contains basic factor moedl framework like calculate regression and factor exposure. New strategies goes here.
./optimization/: portfolio optimization, the extended Markowitz approach.
./output/: saves the program output
./log/: saves console output


