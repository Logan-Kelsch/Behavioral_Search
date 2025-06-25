import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from typing import Tuple, List, Any

def evaluate_forest(
	forest: np.ndarray,
	close_prices: np.ndarray,
	lag_range: Tuple[int, int] = (2, 5),
	n_bins: int = 50
) -> Tuple[pd.DataFrame, List[int], List]:
	"""
	Plot cumulative P&L over time for top signals (with best lagged profit),
	and plot distribution of raw P&L values with zero-centered bins.

	Parameters:
	- forest: 2D array (n_samples, n_trees) of feature values
	- close_prices: 1D array of close prices
	- lag_range: tuple (min_lag, max_lag) to search for best profit lag
	- n_bins: number of bins for PnL distribution (must be even for 0 on edge)

	Returns:
	- scores_df: DataFrame of metrics (ic, profit, r2, infrequency, lnpl, combined)
	- feature_idx_list: unique list of feature indices sorted by combined score
	"""
	# 1) compute returns and market curve
	returns = pd.Series(close_prices).pct_change().fillna(0)
	market_cum = returns.cumsum()

	# 2) precompute lagged returns for IC
	min_lag, max_lag = lag_range
	returns_lagged = {lag: returns.shift(-lag) for lag in range(min_lag, max_lag+1)}

	# 3) prepare features
	n_samples, n_trees = forest.shape
	feature_names = [f"feat_{i}" for i in range(n_trees)]
	df_feats = pd.DataFrame(forest, columns=feature_names)

	cum_pnl_dict = {}
	raw_pnl_dict = {}
	scores = {}

	# 4) compute signals, best-lag profit, and metrics
	for col in feature_names:
		feat = df_feats[col]
		if feat.isna().all() or (feat.fillna(0) == 0).all():
			continue

		signals = {
			'>mean': (feat > feat.mean()).astype(int),
			'<mean': (feat < feat.mean()).astype(int),
			'>0':    (feat > 0).astype(int),
			'<0':    (feat < 0).astype(int),
		}

		for label, sig in signals.items():
			if sig.nunique() <= 1:
				continue

			# find best lag for profit within lag_range
			best_profit = None
			best_lag = None
			for lag in range(min_lag, max_lag+1):
				exec_sig_k = sig.shift(lag).fillna(0)
				profit_k = (exec_sig_k * returns).sum()
				if best_profit is None or profit_k > best_profit:
					best_profit = profit_k
					best_lag = lag

			# compute with best lag
			exec_sig = sig.shift(best_lag).fillna(0)
			raw_pnl = exec_sig * returns
			cum_pnl = raw_pnl.cumsum()

			key = f"{col}_{label}[{best_lag}]"
			raw_pnl_dict[key] = raw_pnl
			cum_pnl_dict[key] = cum_pnl

			# Information Coefficient
			ic = max(abs(returns_lagged[lag].corr(feat)) for lag in returns_lagged)

			total_profit = raw_pnl.sum()
			# consistency (R^2)
			time_idx = np.arange(len(cum_pnl)).reshape(-1,1)
			lr = LinearRegression().fit(time_idx, cum_pnl.values)
			consistency = r2_score(cum_pnl.values, lr.predict(time_idx))

			# infrequency
			infrequency = np.sqrt(1.0 - exec_sig.sum()/n_samples)

			# lnpl
			sum_pos = raw_pnl[raw_pnl>0].sum()
			sum_neg = -raw_pnl[raw_pnl<0].sum()
			if sum_neg>0:
				ratio = sum_pos/sum_neg if sum_pos>0 else 1.0
			else:
				ratio = 0.1
			lnpl = np.log(ratio) if ratio>0 else -np.inf

			# combined
			combined = total_profit * consistency * infrequency * ic * lnpl
			neg_count = sum([total_profit<0, consistency<0, lnpl<0])
			if neg_count>=2:
				combined = -abs(combined)

			scores[key] = {
				'ic': ic,
				'profit': total_profit,
				'r2': consistency,
				'sqrtinfq': infrequency,
				'lnpl': lnpl,
				'combined': combined
			}

	# 5) build DataFrames
	pnls = pd.concat(cum_pnl_dict, axis=1)
	scores_df = pd.DataFrame(scores).T.sort_values('combined', ascending=False)

	# unique feature index list
	feature_idx_list = []
	eval_score_list  = []
	for key in scores_df.index:
		combined_score = scores_df.loc[key, 'combined']
		if(combined_score>0):
			idx = int(key.split('_')[1])
			if (idx not in feature_idx_list):
				feature_idx_list.append(idx)
				eval_score_list.append(combined_score)

	# 6) identify best signals
	best_signals = {
		'Best IC': scores_df['ic'].idxmax(),
		'Best Profit': scores_df['profit'].idxmax(),
		'Best R2': scores_df['r2'].idxmax(),
		'Best Combined': scores_df['combined'].idxmax(),
	}

	# 7) plot cumulative PnL
	plt.figure(figsize=(12,6))
	plt.plot(pnls.index, market_cum, color='black', label='Market')
	for title, sig_key in best_signals.items():
		plt.plot(pnls.index, pnls[sig_key], label=f"{title} ({sig_key})")
	plt.legend()
	plt.xlabel("Time")
	plt.ylabel("Cumulative P&L")
	plt.title("Cumulative P&L: Market vs. Top Signals")
	plt.tight_layout()
	plt.show()

	# 8) distribution with zero-centered bins
	all_top = np.concatenate([raw_pnl_dict[k].values for k in best_signals.values()])
	M = np.max(np.abs(all_top))
	bins = np.linspace(-M, M, n_bins+1)

	plt.figure(figsize=(10,6))
	for title, sig_key in best_signals.items():
		data = raw_pnl_dict[sig_key].dropna()
		plt.hist(data, bins=bins, density=True, histtype='step', label=title)
	plt.axvline(0, color='black', linewidth=1)
	plt.legend()
	plt.xlim((-0.01,0.01))
	plt.xlabel("PnL per period")
	plt.ylabel("Density")
	plt.title("Distribution of PnL for Top Signals (Zero-Centered)")
	plt.tight_layout()
	plt.show()

	return scores_df, feature_idx_list, eval_score_list