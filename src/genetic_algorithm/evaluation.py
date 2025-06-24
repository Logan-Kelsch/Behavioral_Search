

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

def plot_best_feature_pnls(
	forest: np.ndarray,
	close_prices: np.ndarray,
	max_lag: int = 5
):
	"""
	Plot cumulative P&L over time for:
	  - Feature signals: over/under mean and over/under zero (non-constant only)
	  - Market cumulative return in black

	Signals that are constant (all True or all False) or from forest all zeros/NaNs are disqualified.

	Returns:
		scores_df: DataFrame of scores sorted by combined score
		feature_idx_list: list of feature indices (ints) sorted by combined score,
						  extracted from signal keys by stripping 'feat_' and suffixes
	"""
	# Compute returns and market curve
	returns = pd.Series(close_prices).pct_change().fillna(0)
	market_cum = returns.cumsum()

	# Prepare feature DataFrame
	n_samples, n_trees = forest.shape
	feature_names = [f"feat_{i}" for i in range(n_trees)]
	df_feats = pd.DataFrame(forest, columns=feature_names)

	pnls = pd.DataFrame(index=returns.index)
	scores = {}

	for col in feature_names:
		feat = df_feats[col]
		# disqualify entire feature if all zeros or NaNs
		if feat.isna().all() or (feat.fillna(0) == 0).all():
			continue

		signals = {
			'>mean': (feat > feat.mean()).astype(int),
			'<mean': (feat < feat.mean()).astype(int),
			'>0': (feat > 0).astype(int),
			'<0': (feat < 0).astype(int)
		}

		for label, sig in signals.items():
			# disqualify constant signals
			if sig.nunique() <= 1:
				continue

			key = f"{col}_{label}"
			pnl = sig.shift(1).fillna(0) * returns
			cum_pnl = pnl.cumsum()
			pnls[key] = cum_pnl

			# Information Coefficient
			ic = max(
				abs(returns.shift(-lag).corr(feat))
				for lag in range(1, max_lag + 1)
			)
			# Total Profit
			total_profit = pnl.sum()
			# Consistency: R² of cum PnL vs its linear trend
			time = np.arange(len(cum_pnl)).reshape(-1, 1)
			lr = LinearRegression().fit(time, cum_pnl.values)
			trend_line = lr.predict(time)
			consistency = r2_score(cum_pnl.values, trend_line)

			scores[key] = {
				'ic': ic,
				'profit': total_profit,
				'r2': consistency,
				'combined': total_profit * consistency
			}

	# Build sorted scores DataFrame
	scores_df = pd.DataFrame(scores).T.sort_values('combined', ascending=False)

	# Extract ordered list of feature indices
	feature_idx_list = [int(key.split('_')[1]) for key in scores_df.index]
	best_tree_list = []
	for i in feature_idx_list:
		if(i not in best_tree_list):
			best_tree_list.append(i)

	# Identify best signals
	best = {
		'Best IC': scores_df['ic'].idxmax(),
		'Best Profit': scores_df['profit'].idxmax(),
		'Best R2': scores_df['r2'].idxmax(),
		'Best Combined': scores_df['combined'].idxmax()
	}

	# Plot
	plt.figure()
	plt.plot(pnls.index, market_cum, color='black', label='Market')
	for title, sig_key in best.items():
		plt.plot(pnls.index, pnls[sig_key], label=f"{title} ({sig_key})")
	plt.legend()
	plt.xlabel("Time")
	plt.ylabel("Cumulative P&L")
	plt.title("Cumulative P&L: Market vs. Top Feature Signals")
	plt.show()

	return scores_df, best_tree_list