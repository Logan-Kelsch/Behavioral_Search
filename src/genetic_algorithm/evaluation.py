import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from typing import Tuple, List, Any
import utility as utility
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from typing import Tuple, List, Any
import visualization as visualization


def evaluate_forest_newer(
	forest: np.ndarray,
	close_prices: np.ndarray,
	lag_range: Tuple[int, int] = (1, 5)
):
	"""
	For each feature in `forest` and each simple threshold signal,
	finds the lag in lag_range that maximizes the absolute
	information coefficient with the forward-n return:
		(price[t+lag] / price[t]) - 1
	Returns:
	  - scores_df: DataFrame with index "feat_i_label[lag=k]" and column "ic"
	  - feature_idx_list: unique feature indices in the order they first appear
	  - eval_score_list: corresponding IC values (nan→0) for each feature index
	"""
	# 1) price series
	price = pd.Series(close_prices)

	# 2) unpack lags
	min_lag, max_lag = lag_range

	# 3) precompute forward-n returns for each lag
	forward_returns = {
		lag: (price.shift(-lag) / price - 1).fillna(0)
		for lag in range(min_lag, max_lag + 1)
	}

	# 4) prepare features
	feature_names = [f"feat_{i}" for i in range(forest.shape[1])]
	df_feats = pd.DataFrame(forest, columns=feature_names).fillna(0)

	# 5) compute best‐lag IC for every feature+signal
	ic_scores = {}
	for col in feature_names:
		feat = df_feats[col]

		signals = {
			'>mean': (feat > feat.mean()).astype(int),
			'<mean': (feat < feat.mean()).astype(int),
			'>0':    (feat > 0).astype(int),
			'<0':    (feat < 0).astype(int),
			'raw':	 (feat).astype(np.float32)
		}

		for label, sig in signals.items():
			best_ic = None
			best_lag = None

			for lag, ret in forward_returns.items():
				#print(f'LOGIC CHECK PRINTOUT LOGIC CHECK PRINTOUT')
				#print(np.isnan(sig).any(), np.isinf(sig).any())
				#print(np.isnan(ret).any(), np.isinf(ret).any())
				#print(np.nanstd(sig), np.nanstd(ret))

				#ic = sig.corr(ret)
				
				ic = utility.safe_corr(sig, ret)
				
				if best_ic is None or abs(ic) > abs(best_ic):
					best_ic = ic
					best_lag = lag

			key = f"{col}_{label}[lag={best_lag}]"
			ic_scores[key] = best_ic

	# 6) build scores_df
	scores_df = pd.Series(ic_scores, name='ic').to_frame()

	# 7) unique feature index list + eval_score_list
	feature_idx_list = []
	eval_score_list  = []
	for key in scores_df.index:
		ic_val = scores_df.loc[key, 'ic']
		idx = int(key.split('_')[1])
		#print(f'idx: {idx}')
		if idx not in feature_idx_list:
			feature_idx_list.append(idx)
			if(np.isnan(ic_val)):
				print('NAN IC FOUND NAN IC FOUND!!!')
			eval_score_list.append(0 if np.isnan(ic_val) else ic_val)

	return scores_df, feature_idx_list, eval_score_list



'''
def evaluate_forest(
	forest: np.ndarray,
	close_prices: np.ndarray,
	lag_range: Tuple[int, int] = (2, 5),
	n_bins: int = 50,
	plot_eval	:	bool	=	False
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
	print('ENTERING WRONG FUNCTION!!!!!!')
	# 1) compute returns and market curve
	returns = pd.Series(close_prices).pct_change().fillna(0)
	market_cum = returns.cumsum()

	# 2) precompute lagged returns for IC
	min_lag, max_lag = lag_range
	returns_lagged = {lag: returns.shift(-lag) for lag in range(min_lag, max_lag+1)}

	# 3) prepare features
	n_samples, n_trees = forest.shape

	#print(f'in eval, n_trees:{n_trees}')
	feature_names = [f"feat_{i}" for i in range(n_trees)]
	df_feats = pd.DataFrame(forest, columns=feature_names)

	cum_pnl_dict = {}
	raw_pnl_dict = {}
	scores = {}

	#print(f'num feature names in eval: {len(feature_names)}')

	# 4) compute signals, best-lag profit, and metrics
	for col in feature_names:
		feat = df_feats[col]
		#if feat.isna().all() or (feat.fillna(0) == 0).all():
		#	continue
		feat.fillna(0)

		signals = {
			'>mean': (feat > feat.mean()).astype(int),
			'<mean': (feat < feat.mean()).astype(int),
			'>0':    (feat > 0).astype(int),
			'<0':    (feat < 0).astype(int),
		}

		#print(f'num signals inloop in eval: {len(signals)}')

		for label, sig in signals.items():
			#if sig.nunique() <= 1:
			#	continue

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

			#information Coefficient
			print(returns_lagged.keys())
			for lag in returns_lagged:
				print(lag)
				print(returns_lagged[lag].corr(feat))
			ic = max(abs(returns_lagged[lag].corr(feat)) for lag in returns_lagged)

			total_profit = raw_pnl.sum()
			#consistency (R^2)
			time_idx = np.arange(len(cum_pnl)).reshape(-1,1)
			lr = LinearRegression().fit(time_idx, cum_pnl.values)
			consistency = r2_score(cum_pnl.values, lr.predict(time_idx))

			#lnpl
			sum_pos = raw_pnl[raw_pnl>0].sum()
			sum_neg = -raw_pnl[raw_pnl<=0].sum()
			if sum_neg>0:
				ratio = sum_pos/sum_neg if sum_pos>0 else 0.1
			else:
				ratio = 0.1
			lnpl = min(ratio,2)

			# combined
			combined = total_profit * consistency * ic * lnpl
			neg_count = sum([total_profit<0, consistency<0, lnpl<0])
			if neg_count>=2:
				combined = -abs(combined)

			scores[key] = {
				'ic': ic,
				'profit': total_profit,
				'r2': consistency,
				'lnpl': lnpl,
				'combined': combined
			}

	# 5) build DataFrames
	pnls = pd.concat(cum_pnl_dict, axis=1)
	scores_df = pd.DataFrame(scores).T#.sort_values('combined', ascending=False)

	# unique feature index list
	feature_idx_list = []
	eval_score_list  = []
	for key in scores_df.index:
		combined_score = scores_df.loc[key, 'combined']
		idx = int(key.split('_')[1])
		if (idx not in feature_idx_list):
			feature_idx_list.append(idx)
			if(np.isnan(combined_score)):
				eval_score_list.append(0)
			else:
				eval_score_list.append(combined_score)

	# 6) identify best signals
	best_signals = {
		'Best IC': scores_df['ic'].idxmax(),
		'Best Profit': scores_df['profit'].idxmax(),
		'Best R2': scores_df['r2'].idxmax(),
		'Best Combined': scores_df['combined'].idxmax(),
	}

	if(plot_eval):
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
'''



def get_best_forest(
	forfeat_batches	:	list,
	forest_batches	:	list,
	prll_idx_batches:	list,
	close_prices	:	np.ndarray,
	lag_range		:	Tuple[int,int] = (2, 5)
):
	
	scores_batches = []

	#print(f'forfeat forest type: {type(forfeat_batches[0])}')

	for forest in forfeat_batches:
		_, __, local_scores = evaluate_forest_newer(forest,close_prices,lag_range)
		scores_batches.append(local_scores)

	forest_size = len(forest_batches[0])

	best_scores = [float('-inf') for _ in range(forest_size)]
	best_forest = forest_batches[0]

	#use nested zips to update bests according to batch info
	for scores, idxs, trees in zip(scores_batches, prll_idx_batches, forest_batches):
		for s, idx, tree in zip(scores, idxs, trees):
			if(s>best_scores[idx]):
				best_scores[idx] = s
				best_forest[idx] = tree

	#return the forest consisting of the best tree from each prll_idx value, found at each forest[i] location
	#also return the scores from each

	return best_forest, best_scores


from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

def standard_NN_construction(X_train, y_train, epochs=250):
	import tensorflow as tf
	from keras.optimizers.schedules import ExponentialDecay

	reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
		monitor='val_loss',
		factor=0.84, 
		patience=8, 
		min_lr=1e-6
	)
	early_stopping = EarlyStopping(monitor='loss', patience=25, mode='min', restore_best_weights=True)

	opt  = tf.keras.optimizers.Adam(learning_rate=0.01)
	opt2 = tf.keras.optimizers.SGD(learning_rate=0.01)

	def build_model():
		model = tf.keras.Sequential([
			tf.keras.layers.Input(shape=(X_train.shape[1],)),
			tf.keras.layers.Dense(64, activation='linear'),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.Dense(64, activation='linear'),  
			tf.keras.layers.Dropout(0.3),
			tf.keras.layers.Dense(64, activation='linear'),       
			tf.keras.layers.Dense(1, activation='linear')  # Output layer for regression
		])
		
		rmse='root_mean_squared_error'

		model.compile(optimizer=opt2, loss='mse', metrics=['R2Score'])
		return model

	with tf.device('/GPU:0'):
		model = build_model()
		history = model.fit(X_train, y_train, epochs=epochs, batch_size=256, \
						validation_split=0.2, verbose=0, shuffle=False, callbacks=[reduce_lr, early_stopping])
		
	return model, history

def standard_NN_evaluation(
	X_train,X_test,y_train,y_test,
	model,
	history,
	run_dir,
	vizout
):
	y_pred = model.predict(X_test)
	y_pred_train = model.predict(X_train)

	self_r2, self_qacc = visualization.visualize_regression_eval(y_test=y_train, y_pred=y_pred_train, title='Self Test', run_dir=run_dir)
	ind_r2, ind_qacc = visualization.visualize_regression_eval(y_test=y_test, y_pred=y_pred, title='Independent Test', run_dir=run_dir)

	loss = history.history['loss'][1:]
	val_loss = history.history.get('val_loss', [])[1:]
	lr = history.history['learning_rate'][1:]
	epochs = range(2, len(loss) + 2)  # since we sliced off the first

	fig, ax1 = plt.subplots(figsize=(9, 6))

	# Plot loss on left y‐axis
	ax1.plot(epochs, loss,   label='Train Loss',      color='black')
	if val_loss:
		ax1.plot(epochs, val_loss, label='Validation Loss', color='red')
	ax1.set_yscale('log')
	ax1.set_xlabel('Epoch')
	ax1.set_ylabel('Loss')
	ax1.legend(loc='upper right')

	# Create a second y‐axis sharing the same x
	ax2 = ax1.twinx()
	ax2.plot(epochs, lr, label='Learning Rate', color='gray', linestyle='--')
	ax2.set_ylabel('Learning Rate')
	ax2.legend(loc='upper right')

	plt.title('Loss & Learning Rate over Epochs')
	h1, l1 = ax1.get_legend_handles_labels()
	h2, l2 = ax2.get_legend_handles_labels()

	# combine them and draw one legend on ax1:
	ax1.legend(h1 + h2, l1 + l2, loc='upper right', ncol=3)


	#plt.tight_layout()
	#plt.show()
	plt.tight_layout()

	if(vizout):
		fig.savefig(str(run_dir / 'training.png'))

	plt.close()

	loss_NN = 1 - (np.clip(self_r2 * np.sqrt(self_qacc), 0, 1) * np.clip(np.sign(ind_r2)*(ind_r2 * np.sqrt(ind_qacc))**2, 0, 1))

	del fig, ax1, ax2

	return loss_NN