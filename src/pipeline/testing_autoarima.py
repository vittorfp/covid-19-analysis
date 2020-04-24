# -*- coding: utf-8 -*-
__author__ = 'Marco Borges'
__version__ = '0.1.0'

from pathlib import Path
import datetime as dt
import numpy as np
import pandas as pd
import pmdarima as pm

p_data = Path(__file__).resolve().parents[2].joinpath('data', 'raw', 'arquivo_geral.csv')
p_rep = Path(__file__).resolve().parents[2].joinpath('reports')

def main():
	if not p_data.is_file():
		raise FileNotFoundError('Must provide a raw data!')

	df = pd.read_csv(str(p_data), sep=';', parse_dates=['data'])

	new_cases = df.groupby(['data']).agg({
		'casosNovos': 'sum'
	})

	time_series = new_cases.copy()

	first_100 = np.where(time_series.cumsum() >= 100)[0][0]
	time_series = time_series[first_100:]
	dtime_series = pd.Series(np.gradient(time_series['casosNovos'].values), index=time_series.index)

	# stepwise_fit = pm.auto_arima(
	# 	dtime_series, start_p=1, start_q=1, max_p=15, max_q=15,
	# 	seasonal=False, d=0, trace=True, error_action='ignore',
	# 	suppress_warnings=True, stepwise=True, n_jobs=-1
	# )
	summ = ''
	aic = 1e6
	for m in range(1, 15):
		d = pm.arima.ndiffs(time_series)
		if m > 1:
			try:
				D = pm.arima.nsdiffs(time_series, m)
			except:
				D = 1
		else:
			D = 0
		stepwise_fit = pm.auto_arima(
			time_series, start_p=1, start_q=1, max_p=15, max_q=15, m=m,
			start_P=0, seasonal=True, d=d, D=D, trace=True, error_action='ignore',
			suppress_warnings=True, stepwise=True, n_jobs=-1
		)
		if stepwise_fit.aic() < aic:
			aic = stepwise_fit.aic()
			summ = stepwise_fit.summary()

	now = dt.datetime.now()
	filename = now.strftime('%Y-%m-%d_%H-%M-%S') + '_report.csv'
	filepath = p_rep.joinpath(filename)
	with open(str(filepath), 'w') as f:
		f.write(summ.as_csv())
	print(summ)

if __name__ == '__main__':
	main()