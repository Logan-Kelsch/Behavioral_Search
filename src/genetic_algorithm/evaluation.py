import numpy as np
import bottleneck as bn
from numba import njit, prange
import pandas as pd
from sklearn.metrics import f1_score
from typing import Tuple
import utility
import matplotlib.pyplot as plt
import visualization
import math

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


def generate_pl_atrplr(
	price_data	:	np.ndarray,
	atr_thresh	:	float,
	ln_plratio	:	float,
	direction	:	int	=	1,
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
	vect_exit = _exit_pl_kernel(
		price_data[:, 1], price_data[:, 2], price_data[:, 3],
		vect_atr, up_atr, dn_atr
	)

	return vect_exit

def generate_solarr_atrplr(
	price_data	:	np.ndarray,
	atr_thresh	:	float,
	ln_plratio	:	float,
	direction	:	int	=	1,
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

def generate_solarr_atrplr_asatr(
	price_data	:	np.ndarray,
	atr_thresh	:	float,
	ln_plratio	:	float,
	direction	:	int	=	1,
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

	pos = np.where(vect_exit==1)
	neg = np.where(vect_exit==0)
	
	vect_exit[pos] =  vect_atr[pos] * up_atr
	vect_exit[neg] = -vect_atr[neg] * dn_atr

	return vect_exit


#using numba to optimize these loops
@njit
def _exit_signals_kernel(high, low, close, atr, up_atr, dn_atr):
	N = high.shape[0]
	out = np.empty(N, np.float64)
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


@njit
def _exit_pl_kernel(high, low, close, atr, up_atr, dn_atr):
	"""
	For each entry at close[i], compute the profit or loss (PL) of the trade if
	it exits on either the profit target or stop-loss threshold, both defined
	as multiples of ATR. If both thresholds are hit in the same bar, or if neither
	is hit by the end of the series, returns np.nan for that entry.

	Parameters
	----------
	high : 1d array of float64
		Series of high prices.
	low : 1d array of float64
		Series of low prices.
	close : 1d array of float64
		Series of close prices (entry prices).
	atr : 1d array of float64
		Series of average true range values.
	up_atr : float64
		Multiplier for the profit-target ATR (profit threshold = close + up_atr * atr).
	dn_atr : float64
		Multiplier for the stop-loss ATR (stop threshold = close - dn_atr * atr).

	Returns
	-------
	pl : 1d array of float64
		Profit (positive) or loss (negative) for each entry; np.nan if ambiguous
		(both thresholds hit in the same bar) or never hit.
	"""
	N = high.shape[0]
	pl = np.empty(N, np.float64)
	# initialize all outputs to NaN
	pl[:] = np.nan

	for i in range(N):
		entry_price = close[i]
		up_thr = entry_price + up_atr * atr[i]
		dn_thr = entry_price - dn_atr * atr[i]

		# scan forward until one of the thresholds is triggered
		for j in range(i + 1, N):
			hi = high[j]
			lo = low[j]

			# ambiguous: both profit and stop‐loss hit in same bar
			if hi >= up_thr and lo <= dn_thr:
				pl[i] = np.nan
				break
			# profit target first
			elif hi >= up_thr:
				pl[i] = up_thr - entry_price
				break
			# stop‐loss first
			elif lo <= dn_thr:
				pl[i] = dn_thr - entry_price
				break

		# if loop completes without break, pl[i] stays NaN

	return pl

import numpy as np
from numba import njit, prange

@njit(parallel=True)
def _best_precision_scores_numba(
	forest:      np.ndarray,
	y_true:      np.ndarray,
	valid_idx:   np.ndarray
) -> np.ndarray:
	"""
	Numba-accelerated computation of best precision per feature.
	forest:    (n_samples, n_features) array of feature values
	y_true:    (m_valid,) int8 array of ground truth labels
	valid_idx: (m_valid,) int64 array of sample indices corresponding to y_true
	Returns:
		best_precisions: (n_features,) array where each entry is the highest
						 precision (tp / (tp + fp)) achievable by thresholding
						 the feature either >0 or <0.
	"""
	_, n_features = forest.shape
	best_precisions = np.zeros(n_features, dtype=np.float64)

	for j in prange(n_features):
		best = 0.0
		# try both sign-based thresholds
		for pattern in range(2):
			tp = 0
			fp = 0

			# accumulate tp and fp over valid samples
			for idx_k, t in enumerate(valid_idx):
				val = forest[t, j]
				# pattern 0: predict 1 if val > 0; pattern 1: predict 1 if val < 0
				if (pattern == 0 and val > 0.0) or (pattern == 1 and val < 0.0):
					# predicted positive
					if y_true[idx_k] == 1:
						tp += 1
					else:
						fp += 1
				# negatives do not affect precision numerator/denominator

			# compute precision = tp / (tp + fp)
			denom = tp + fp
			prec = tp / denom if denom > 0 else 0.0

			if prec > best:
				best = prec

		best_precisions[j] = best

	return best_precisions


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

		#evaluate 2 simple binary encodings
		best = 0.0
		for pattern in range(2):
			tp = 0
			fp = 0
			fn = 0

			# loop over valid samples only
			for idx_k, t in enumerate(valid_idx):
				val = forest[t, j]
				y = y_true[idx_k]

				# determine prediction
				if pattern == 0:
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

			#computing a custom harmonic mean
			#where precision 2x as important as recall
			denom = 3*tp + 2*fp + fn
			f1 = 3*tp / denom if denom > 0 else 0.0
			if f1 > best:
				best = f1

		best_scores[j] = best

	return best_scores

@njit(parallel=True)
def _best_f1_scores_numba_usemean(
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


from numba import njit, prange
import numpy as np
import math


from numba import njit, prange
import numpy as np
import math

@njit(parallel=True)
def _prec_inconsistency_numba(
	preds: np.ndarray,
	y_true: np.ndarray,
	valid_idx: np.ndarray,
	n_chunks: int,
	eps: float = 1e-9
) -> np.ndarray:
	"""
	Numba‐accelerated inconsistency score for per-feature binary predictions,
	restricted to valid_idx, split into n_chunks. Returns an (n_features,) array
	in [0,1], where 0 = perfectly consistent, 1 = perfectly inconsistent.
	"""
	m_valid = valid_idx.shape[0]
	n_features = preds.shape[1]
	inconsistency = np.empty(n_features, dtype=np.float64)

	# how many samples per chunk
	base = m_valid // n_chunks
	rem  = m_valid - base * n_chunks

	for f in prange(n_features):
		sum_p  = 0.0     # Σ precision
		sum_p2 = 0.0     # Σ precision^2
		cnt    = 0.0     # how many bins had ≥1 prediction
		start  = 0

		for c in range(n_chunks):
			size = base + (1 if c < rem else 0)
			end  = start + size

			tp = 0.0
			fp = 0.0
			for k in range(start, end):
				idx = valid_idx[k]
				pred = preds[idx, f]
				# skip NaNs (works for floats & ints)
				if pred != pred:
					continue
				if pred == 1.0:
					if y_true[k] == 1.0:
						tp += 1.0
					else:
						fp += 1.0

			denom = tp + fp
			if denom > 0.0:
				p = tp / denom
				sum_p  += p
				sum_p2 += p * p
				cnt    += 1.0

			start = end

		if cnt > 0.0:
			# mean precision μ
			mp = sum_p / cnt
			# variance = E[p^2] - μ^2
			var = (sum_p2 / cnt) - (mp * mp)
			sd  = math.sqrt(var) if var > 0.0 else 0.0
			# inconsistency = (σ + eps) / (μ + σ + eps)
			inconsistency[f] = (sd + eps) / (mp + sd + eps)
		else:
			# never predicted ⇒ treat as maximally inconsistent
			inconsistency[f] = 1.0

	return inconsistency

@njit(parallel=True, fastmath=True, cache=True)
def _best_precision_per_feature(X, y):
	"""
	X: float64 (n_samples, n_features)
	y: uint8 (n_samples,)  values in {0,1}
	Returns:
	  f_ps: float64 (n_features,)  best precision per feature
	  flip_idx: uint8 (n_features,) 1 if best is (X[:,j] < 0), else 0 for (X[:,j] > 0)
	"""
	n_samples, n_features = X.shape
	f_ps = np.zeros(n_features, dtype=np.float64)
	flip_idx = np.zeros(n_features, dtype=np.uint8)

	for j in prange(n_features):
		tp_pos = 0
		fp_pos = 0
		tp_neg = 0
		fp_neg = 0

		for i in range(n_samples):
			x = X[i, j]
			yi = y[i]

			# x > 0 → positive-branch prediction
			if x > 0.0:
				if yi == 1:
					tp_pos += 1
				else:
					fp_pos += 1
			# x < 0 → negative-branch prediction
			elif x < 0.0:
				if yi == 1:
					tp_neg += 1
				else:
					fp_neg += 1
			# x == 0 (or NaN) → no prediction; skip

		denom_pos = tp_pos + fp_pos
		denom_neg = tp_neg + fp_neg

		prec_pos = (tp_pos / denom_pos) if denom_pos > 0 else 0.0
		prec_neg = (tp_neg / denom_neg) if denom_neg > 0 else 0.0

		# pick the better of the two; ties favor (>0) branch (flip_idx=0)
		if prec_neg > prec_pos:
			f_ps[j] = prec_neg
			flip_idx[j] = 1  # use X[:,j] < 0
		else:
			f_ps[j] = prec_pos
			flip_idx[j] = 0  # use X[:,j] > 0

	return f_ps, flip_idx

@njit(parallel=True, fastmath=True, cache=True)
def _binarize_by_flips_numba(X, flips_u8):
	n, m = X.shape
	out = np.empty((n, m), dtype=np.uint8)
	for j in prange(m):
		use_pos = (flips_u8[j] == 0)
		if use_pos:
			for i in range(n):
				x = X[i, j]
				out[i, j] = 1 if x > 0.0 else 0
		else:
			for i in range(n):
				x = X[i, j]
				out[i, j] = 1 if x < 0.0 else 0
	return out

def binarize_features(X, y):
	import evaluation
	ps, flips = evaluation._best_precision_per_feature(X, y)

	Xc = np.ascontiguousarray(X)
	flips = np.asarray(flips).ravel()
	if(flips.shape[0]!=Xc.shape[1]):
		raise ValueError('mismatching sizes: flips to Xc in coef calc prep')
	flips_u8 = (flips != 0).astype(np.uint8)

	return evaluation._binarize_by_flips_numba(Xc, flips_u8)

def solve_EV(y_true, y_pred, R):
	ps = np.sum(y_true & y_pred)/np.sum(y_pred) if np.sum(y_pred)>0 else 0.0
	return ((R+1) * ps - 1)

def evaluate_forest_adjev(
	X	:	np.ndarray,
	y	:	np.ndarray,
	R	:	float	=	5,
	balance_precision	:	bool	=	True,
	balance_frequency	:	bool	=	False,
	clip_negative_evs	:	bool	=	True
):
	#identify valid (non-NaN) indices and extract y_true
	#valid_mask = ~np.isnan(y)
	#valid_idx = np.nonzero(valid_mask)[0].astype(np.int64)
	#y = y[valid_mask].astype(np.int8)

	#collect precision scores and trees to flip
	ps, flips = _best_precision_per_feature(X, y)
	
	#solve for EV
	EV = (R+1) * ps - 1
	#print(f'EVS: {EV}')
	if(EV.shape[0]!=0):
		#print(f'MAX EV: {EV.max()}')

		Xc = np.ascontiguousarray(X)
		flips = np.asarray(flips).ravel()
		if(flips.shape[0]!=Xc.shape[1]):
			raise ValueError('mismatching sizes: flips to Xc in coef calc prep')
		flips_u8 = (flips != 0).astype(np.uint8)

		X = _binarize_by_flips_numba(Xc, flips_u8)

		#decay EV if predictions have unreasonable frequency (scarce, excessive)
		if(balance_frequency):
			bic = binary_imbalance_coefficient(X, y)
			#print(bic.min(), bic.max())
			EV *= ((1-bic) ** 2)
		
		#decay EV if inconsistent precision in predictions
		if(balance_precision):
			EV *= ((1-_prec_inconsistency_numba(X, y, np.arange(X.shape[0], dtype=int), n_chunks=4)) ** 2)
			pass

		#print(f'MAX EV: {EV.max()}')

		if(clip_negative_evs):
			np.clip(EV, 0, None, out=EV)

		#print(f'MAX adjEV: {EV.max():.6f}')


	return EV, flips


def evaluate_forest_atrplr( 
	forest	: np.ndarray,
	solarr	: np.ndarray,
	bal_coef: bool = True,
	prc_coef: bool = True
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
	f1s = _best_precision_scores_numba(forest, y_true, valid_idx)

	if(bal_coef):
		f1s:np.ndarray = f1s * (1-binary_imbalance_coefficient(forest[valid_mask], solarr[valid_mask])**2)

	if(prc_coef):

		f1s:np.ndarray = f1s * (1-_prec_inconsistency_numba(forest, y_true, valid_idx, n_chunks=10)**2)

	return f1s
	
def evaluate_tree_atrplr_pl(forest: np.ndarray, pl: np.ndarray):
	"""
	For each feature column in `forest`, evaluate four binary thresholds:
	  1) feature > its mean
	  2) feature < its mean
	  3) feature > 0
	  4) feature < 0
	Multiply each mask by `pl`, sum the result (ignoring NaNs), and select
	the mask giving the highest sum. Preserves NaNs from `pl`.
	
	Parameters
	----------
	forest : np.ndarray, shape (n_samples, n_features)
		Your candidate‐feature matrix.
	pl : np.ndarray, shape (n_samples,)
		Profit/Loss per instance (may contain NaNs).
	
	Returns
	-------
	best_pls : List[np.ndarray]
		For each feature, the masked PL array under the best threshold.
	best_sums : np.ndarray, shape (n_features,)
		The max sum of masked PL for each feature.
	best_thresholds : List[str]
		Which threshold won for each feature: one of (">mean","<mean",">0","<0").
	"""
	n_samples, n_features = forest.shape
	best_pls = []
	best_sums = np.empty(n_features, dtype=np.float64)
	best_thresholds = []
	
	for j in range(n_features):
		feat = forest[:, j]
		mean = np.nanmean(feat)
		
		# build masks and names
		masks = [
			feat > mean,
			feat < mean,
			feat > 0,
			feat < 0,
		]
		names = [">mean", "<mean", ">0", "<0"]
		
		# track best
		best_sum = -np.inf
		best_pl = None
		best_name = None
		
		for mask, name in zip(masks, names):
			# cast to 0/1 float mask
			m = mask.astype(np.float64)
			masked_pl = pl * m
			s = np.nansum(masked_pl)
			if s > best_sum:
				best_sum = s
				best_pl = masked_pl
				best_name = name
		
		best_pls.append(best_pl)
		best_sums[j] = best_sum
		best_thresholds.append(best_name)
	
	return best_pls, best_sums, best_thresholds


	

def binary_imbalance_coefficient(
	forest:  np.ndarray,
	sol_arr:  np.ndarray
) -> np.ndarray:
	"""
	This function is used to create a coefficient representing
	desirable punishment for unreasonable market participation.
	This function considers overparticipation (more frequent than solution)
	and considers underparticipation (subject to test-data anomalies)

	Flat-bottom imbalance loss, more efficiently:
	  loss = |p_feats - clip(p_feats, lower, p_sol)|

	Any p_feats in [lower, p_sol] → zero; 
	below → lower - p_feats; above → p_feats - p_sol.
	"""
	# flatten solution and sanity‐check
	sol = sol_arr.ravel()
	if forest.shape[0] != sol.shape[0]:
		raise ValueError(f"Sample mismatch: {forest.shape[0]} vs {sol.shape[0]}")

	# compute rates
	p_sol   = sol.mean()
	lower   = p_sol/10
	p_feats = forest.mean(axis=0)

	# single‐line flat‐bottom: clip then abs diff
	return np.abs(p_feats - np.clip(p_feats, lower, p_sol))

def shapley_ev_allvote(
	X: np.ndarray,           # (n_samples, n_models) in {0,1}
	y_true: np.ndarray,      # (n_samples,) in {0,1}
	R: float = 5.0,
	n_permutations: int = 256,
	seed: int | None = None,
	return_kind: str = "shapley"   # "both", "shapley", or "importance"
):
	"""
	Monte Carlo Shapley values for model contributions to an ALL-vote (AND) ensemble,
	where coalition predictions are 1 iff *all* included models predict 1.
	Score = EV from precision: EV = (R+1)*P - 1.

	Returns
	-------
	If return_kind == "both": dict with keys:
		- "shapley": np.ndarray (n_models,) raw Shapley  contributions (EV units)
		- "importance": np.ndarray (n_models,) normalized to [0,1] (positive part)
		- "ev_full": float, EV of the full coalition (all models)
	If "shapley": np.ndarray (n_models,)
	If "importance": np.ndarray (n_models,)
	"""

	#print(f'in shapley_ev_allvote in evaluation: X, y_ shapes: {X.shape} -- {y_true.shape}')

	if(X.size == 0):
		return np.array([])
	
	valid_mask = ~np.isnan(y_true)
	X = (X[valid_mask].astype(np.uint8) != 0)
	y = (np.asarray(y_true[valid_mask]).astype(np.uint8) != 0)

	rng = np.random.default_rng(seed)
	n_samples, n_models = X.shape

	# ---- helper to compute EV from TP/FP (precision-only) ----
	def ev_from_counts(tp: float, fp: float) -> float:
		denom = tp + fp
		if denom <= 0.0:
			return 0.0
		P = tp / denom
		return P * (R + 1.0) - 1.0

	# Precompute full-coalition EV (AND across all models)
	p_full = X.all(axis=1)
	tp_full = float(np.sum(p_full & y))
	fp_full = float(np.sum(p_full & (~y)))
	ev_full = ev_from_counts(tp_full, fp_full)

	# For incremental updates, we keep the current coalition mask (boolean over samples)
	# and track EV via TP/FP counts for that mask.
	# NOTE: Base coalition (empty set) uses vacuous truth for AND → all True mask.
	# Adding first model j yields mask = X[:, j], as desired.

	shapley_sum = np.zeros(n_models, dtype=np.float64)

	# Optional precomputations to reduce repeated ANDs with y / ~y
	X_and_y    = X & y[:, None]     # shape: (n_samples, n_models)
	X_and_ny   = X & (~y)[:, None]

	for _ in range(n_permutations):
		order = rng.permutation(n_models)

		# start from empty coalition (mask = all True)
		mask = np.ones(n_samples, dtype=bool)
		# counts for empty coalition
		tp_cur = float(np.sum(mask & y))
		fp_cur = float(np.sum(mask & (~y)))
		ev_cur = ev_from_counts(tp_cur, fp_cur)

		for j in order:
			# New mask = old mask AND model j's predictions
			# We need TP/FP for the new mask quickly.
			# TP_next = sum(mask & X[:,j] & y) = sum((mask & y) & X[:,j])
			# FP_next = sum(mask & X[:,j] & ~y) = sum((mask & ~y) & X[:,j])

			# Compute masked sums via boolean indexing (fast in NumPy)
			# First restrict to current mask; then sum X_and_y / X_and_ny columns j
			idx = mask
			tp_next = float(np.sum(X_and_y[idx, j]))
			fp_next = float(np.sum(X_and_ny[idx, j]))

			ev_next = ev_from_counts(tp_next, fp_next)
			marginal = ev_next - ev_cur
			shapley_sum[j] += marginal

			# Advance coalition
			# (mask & X[:, j]) updates in-place efficiently
			mask[idx] &= X[idx, j]
			tp_cur, fp_cur, ev_cur = tp_next, fp_next, ev_next

	shapley = shapley_sum / float(n_permutations)

	# Importance on [0,1]: normalize positive contributions
	pos = np.clip(shapley, 0.0, None)
	imax = pos.max()
	importance = (pos / imax) if imax > 0 else np.zeros_like(pos)

	if return_kind == "shapley":
		return shapley
	if return_kind == "importance":
		return importance
	return {"shapley": shapley, "importance": importance, "ev_full": ev_full}

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

def meta_biconsensus(
	X	:	np.ndarray
):
	return X.all(axis=1).astype(int)

def meta_bipopvote(
	pred_arrs	:	np.ndarray,
	threshold	:	float	=	0.5
):
	'''
	This function returns an array of the popular vote for an array of output model predictions
	'''
	return (pred_arrs.mean(axis=1) >= threshold).astype(int)
	


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