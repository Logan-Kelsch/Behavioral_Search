import numpy as np
import bottleneck as bn
from numba import njit, prange
import pandas as pd
from sklearn.metrics import f1_score
from typing import Tuple
import utility
import matplotlib.pyplot as plt
import visualization


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


def generate_solarr_atrplr(
	price_data	:	np.ndarray,
	atr_thresh	:	float,
	ln_plratio	:	float,
	direction	:	int,
	atr_window	:	int	=	7
)	->	np.ndarray:
	
	'''
	This function first collects the SIMPLE ATR of the data
	The simple ATR is the average High-Low of each candle averaged over past (atr_window) candles.

	This function then searches forward in time for which direction exits first 
	in terms of traditional PL-ratio
	'''

	#first collect the true range vector
	vect_tr = price_data[:, 1] - price_data[:, 2]
	
	#collect the ATR vector and multiply by threshold (ident is 1)
	vect_atr = bn.move_mean(vect_tr, window=atr_window, min_count=1) * atr_thresh

	#collect the up/down thresholds
	#direction comes in as 1 if bullish ratio skewing
	#direction comes in as 0 if bearish ratio skewing
	up_atr = np.exp(ln_plratio) if direction==1 else 1
	dn_atr = np.exp(ln_plratio) if direction==0 else 1
	
	#collect the forward walk exit vector (boolean)
	vect_exit = _exit_signals_kernel(
		price_data[:, 1], price_data[:, 2], price_data[:, 3],
		vect_atr, up_atr, dn_atr
	)

	return vect_exit


#using numba to optimize these loops
@njit
def _exit_signals_kernel(high, low, close, atr, up_atr, dn_atr):
	N = high.shape[0]
	out = np.empty(N, np.float16)
	out[:] = np.nan

	for i in range(N):
		up_thr   = close[i] + up_atr   * atr[i]
		dn_thr = close[i] - dn_atr * atr[i]

		for j in range(i+1, N):
			hi = high[j]
			lo = low[j]
			if hi >= up_thr and lo <= dn_thr:
				out[i] = np.nan
				break
			elif hi >= up_thr:
				out[i] = 1.0
				break
			elif lo <= dn_thr:
				out[i] = 0.0
				break
	return out

@njit(parallel=True)
def _best_f1_scores_numba(
	forest: np.ndarray,
	y_true: np.ndarray,
	valid_idx: np.ndarray
) -> np.ndarray:
	"""
	Numba-accelerated computation of best F1 scores per feature.
	forest: (n_samples, n_features)
	y_true: (m_valid,) int8 array of ground truth labels
	valid_idx: (m_valid,) int64 array of sample indices corresponding to y_true
	"""
	_, n_features = forest.shape
	best_scores = np.zeros(n_features, dtype=np.float64)

	# For each feature (parallelized)
	for j in prange(n_features):
		# 1) compute mean over valid samples
		sum_val = 0.0
		cnt = 0
		for t in valid_idx:
			sum_val += forest[t, j]
			cnt += 1
		mean_val = sum_val / cnt if cnt > 0 else 0.0

		# 2) evaluate 4 simple binary encodings
		best = 0.0
		for pattern in range(4):
			tp = 0
			fp = 0
			fn = 0

			# loop over valid samples only
			for idx_k, t in enumerate(valid_idx):
				val = forest[t, j]
				y = y_true[idx_k]

				# determine prediction
				if pattern == 0:
					pred = 1 if val > mean_val else 0
				elif pattern == 1:
					pred = 1 if val < mean_val else 0
				elif pattern == 2:
					pred = 1 if val > 0.0 else 0
				else:
					pred = 1 if val < 0.0 else 0

				# accumulate confusion matrix
				if pred == 1:
					if y == 1:
						tp += 1
					else:
						fp += 1
				else:
					if y == 1:
						fn += 1

			# compute F1
			denom = 2 * tp + fp + fn
			f1 = 2 * tp / denom if denom > 0 else 0.0
			if f1 > best:
				best = f1

		best_scores[j] = best

	return best_scores

def evaluate_forest_atrplr( 
	forest	: np.ndarray,
	solarr	: np.ndarray
):
	"""
	Compute the best F1 score for each feature in `forest` against a binary solution vector.<br>
	Wrapper that prepares inputs and calls the Numba kernel.

	Parameters
	----------
	forest : np.ndarray, shape (n_samples, n_features)
		Candidate feature matrix.
	solution : np.ndarray, shape (n_samples,)
		Ground-truth labels (0, 1, or np.nan for ignored samples).

	Returns
	-------
	np.ndarray, shape (n_features,)
		Best F1 score for each feature, across 4 simple threshold-based encodings.


	This forest evaluating function takes in a given forest, 
	derives 4 boolean arrays, two about zero and two about its mean.
	This then takes in or generates a boolean solution array determined by
	exact atr exit threshold && sufficient log(pl ratio).

	##### - - NOTE LESS AGGRESSIVE NOTE - - ############# - - NOTE MORE AGGRESSIVE NOTE - - #####

	lower atr_thresh   ->   short-term noise - - higher atr_thresh   ->   long-term noise 
	ln_plratio  <  0   ->   more probable    - - ln_plratio  >   0   ->   less probable	(assuming balance, 0 => perfect random of 50/50)

	The forest is then scored against the solution array (f1-score?)
	These scores are then brought out of the fitness function to evaluate how to move forward in the next iteration of evolution.

	An example use of this function iteratively would be:
	THIS EXAMPLE IS AN ANY-TIME ALGORITHM VERSION:
	1- initial forest is generated
	2- creating an initial solution set that is extremely solvable, for example having .7 f1score survival thresh and ln_plratio < 0
	3- forest is scored using f1-score and compared to a survival threshold. trees with scores<threshold await replacement
	4- a candidate step is made ONLY (makes it an any-time alg) in the aggressive (+) direction of solution space (ln_plratio, atr_thresh)
	5- a new solution set is generated, and is by definition, slightly harder to solve and maximally relevant to the previous solution set.
	6- candidate population is generated by using MRC (MERC with survival based elitism) for deciding source of replacements for dead trees.
	7- loop back to step three or end search.
	"""

	#identify valid (non-NaN) indices and extract y_true
	valid_mask = ~np.isnan(solarr)
	valid_idx = np.nonzero(valid_mask)[0].astype(np.int64)
	y_true = solarr[valid_mask].astype(np.int8)

	#call the JIT-compiled kernel
	return _best_f1_scores_numba(forest, y_true, valid_idx)
	


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

def standard_NN_construction(X_train, y_train, epochs=250, verbose=0):
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
						validation_split=0.2, verbose=verbose, shuffle=False, callbacks=[reduce_lr, early_stopping])
		
	return model, history


def standard_LM_construction(X_train, y_train):
	from sklearn.linear_model import LinearRegression
	from keras.optimizers.schedules import ExponentialDecay

	model = LinearRegression()


	model.fit(X_train, y_train)
	
		
	return model


def standard_LM_evaluation(
	X_train,X_test,y_train,y_test,
	model,
	run_dir,
	vizout,
	show:bool=False
):

	y_pred = model.predict(X_test)
	y_pred_train = model.predict(X_train)

	self_r2, self_qacc = visualization.visualize_regression_eval(y_test=y_train, y_pred=y_pred_train, title='Self Test', run_dir=run_dir, vizout=vizout, show=show)
	ind_r2, ind_qacc = visualization.visualize_regression_eval(y_test=y_test, y_pred=y_pred, title='Independent Test', run_dir=run_dir, vizout=vizout, show=show)

	self_qacc = (self_qacc * 2 - 1)
	ind_qacc = (ind_qacc * 2 - 1)

	loss_LM = (
		(1-self_r2) * (1-self_qacc) * min(1-ind_r2**2, 1) * (1-ind_qacc**2)
	)

	return loss_LM


def standard_NN_evaluation(
	X_train,X_test,y_train,y_test,
	model,
	history,
	run_dir,
	vizout,
	show:bool=False
):
	y_pred = model.predict(X_test)
	y_pred_train = model.predict(X_train)

	self_r2, self_qacc = visualization.visualize_regression_eval(y_test=y_train, y_pred=y_pred_train, title='Self Test', run_dir=run_dir, vizout=vizout, show=show)
	ind_r2, ind_qacc = visualization.visualize_regression_eval(y_test=y_test, y_pred=y_pred, title='Independent Test', run_dir=run_dir, vizout=vizout, show=show)

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

	#if(show):
	#	plt.show()

	plt.close()

	self_qacc = (self_qacc * 2 - 1)
	ind_qacc = (ind_qacc * 2 - 1)

	loss_NN = (
		(1-self_r2) * (1-self_qacc**2) * min(1-ind_r2**3, 1) * (1-ind_qacc**4)
	)

	del fig, ax1, ax2

	return loss_NN