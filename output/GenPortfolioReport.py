__author__ = 'Derek Qi'
# Read portfolio from csv, generates report base on the csv file
# data are collected from Yahoo fiance

import pandas as pd
from yahoo_finance import Share


def GenPortfolioReport(portfolio_file, report_file):
	ptfl_full = pd.read_csv(portfolio_file)
	ptfl_sel = ptfl_full[ptfl_full['Weight'] != 0]

	ticker_list = list(ptfl_sel['Ticker'])

	row_dict_list = []
	# get Name, prev_close, market_cap, PE, avg_daily_volume, from Yahoo finance
	for ticker in ticker_list:
		stock = Share(ticker)
		stock.refresh()

		name = stock.get_name()
		prev_close = stock.get_prev_close()
		avg_daily_volume = stock.get_avg_daily_volume()
		market_cap = stock.get_market_cap()
		pe_ratio = stock.get_price_earnings_ratio()
		last_tradetime = stock.get_trade_datetime()
		row_dict_list.append({'Ticker':ticker, 'Name':name, 'Prev Close':prev_close, 'Avg Daily Volume':avg_daily_volume, 'Market Cap':market_cap, 'PE ratio':pe_ratio, 'Last Tradetime':last_tradetime})

	report = pd.DataFrame(row_dict_list)
	report = pd.merge(ptfl_sel, report, on='Ticker')
	report = report[['Ticker', 'Weight', 'Name', 'Prev Close', 'Avg Daily Volume', 'Market Cap', 'PE ratio', 'Last Tradetime']]
	print(report)
	report.to_csv(report_file)


if __name__ == "__main__":
	portfolio_long_only_dir = 'portfolio_test_long_only.csv'
	report_long_only_dir = 'Portfolio_Report_long_only.csv'
	GenPortfolioReport(portfolio_long_only_dir, report_long_only_dir)
	portfolio_long_short_dir = 'portfolio_test_long_short.csv'
	report_long_short_dir = 'Portfolio_Report_long_short.csv'
	GenPortfolioReport(portfolio_long_short_dir, report_long_short_dir)
	