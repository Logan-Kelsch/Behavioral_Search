from typing import List, Any, Tuple
import numpy as np
import genetic_algorithm.population as population

import genetic_algorithm.optimize as optimize
import genetic_algorithm.utility as utility
import pandas as pd
import genetic_algorithm.visualization as visualization
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import genetic_algorithm.evaluation as evaluation



from typing import List, Any, Tuple

def shortest_common_supersequence(
	seqs: List[List[Any]]
) -> Tuple[List[Any], List[List[int]]]:
	"""
	Greedy heuristic for shortest common supersequence:
	  - At each step, pick the element that matches the most sequences at their current front.
	  - Advance those sequences by one.
	  - Continue until all sequences are consumed.
	Returns:
	  - scs: the greedy supersequence
	  - idx_lists: for each position in scs, the list of sequence indices advanced
	"""
	k = len(seqs)
	lengths = [len(s) for s in seqs]
	pos = [0] * k
	scs: List[Any] = []
	idx_lists: List[List[int]] = []

	while not all(pos[i] >= lengths[i] for i in range(k)):
		# Gather candidates and which sequences they match
		candidates = {}
		for i in range(k):
			if pos[i] < lengths[i]:
				e = seqs[i][pos[i]]
				candidates.setdefault(e, []).append(i)

		# Pick the candidate covering the most sequences
		# Tie-break by choosing the candidate with smallest repr
		best_e = max(
			candidates.items(),
			key=lambda item: (len(item[1]), -ord(str(item[0])[0]) if isinstance(item[0], str) else len(item[1]))
		)[0]

		# Advance sequences matching best_e
		advanced = []
		for i in range(k):
			if pos[i] < lengths[i] and seqs[i][pos[i]] == best_e:
				pos[i] += 1
				advanced.append(i)

		scs.append(best_e)
		idx_lists.append(advanced)

	return scs, idx_lists


def quickfix_score_to_loss(
	scores  :   list
)   ->  list:
	
	losses = []

	for score in scores:

		losses.append( np.exp( 1 - np.exp(score) ) )

	return losses

def pinder_resize(
	val     :   float,
	pinder  :   float,
	space   :   tuple
):
	
	size_space = space[1]-space[0]

	if(space[1]<space[0]):
		raise ValueError(f"Defined space for pinder resize is not logical. Got [{space[0]},{space[1]}]")

	p_range = pinder*size_space

	under = val - p_range
	over = val + p_range

	#first check if under needs to be passed to over
	if(under<space[0]):
		over += (space[0]-under)
		under = space[0]

	if(over>space[1]):
		under += (over-space[1])
		over = space[1]

	if((under<space[0]) or (over>space[1])):
		raise(f"pinder resize failed! check code."
			  f"val:{val},pinder:{pinder},space:{space}")
	
	return over, under


import re
from pathlib import Path

def fetch_new_run_dirpath():
	runs_dir = Path('../runs')
	runs_dir.mkdir(exist_ok=True)
	pattern = re.compile(r'run_(\d+)$')
	run_nums = []
	for p in runs_dir.iterdir():
		if p.is_dir():
			m = pattern.match(p.name)
			if m:
				run_nums.append(int(m.group(1)))
	next_run = max(run_nums, default=0) + 1
	new_run_dir = runs_dir / f'run_{next_run}'
	new_run_dir.mkdir()	
	return new_run_dir


def demo_constopt_nn(
	pop	=	None
):
	import genetic_algorithm.transforms as transforms
	data = pd.read_csv("../data/ES15.csv")
	x_raw = data.values

	dirpath = utility.fetch_new_run_dirpath()

	if(pop==None):
		pop = population.generate_random_forest(200, 5)

	np.seterr(all='ignore')

	best_forest, best_scores, best_overtime = optimize.optimize_constants(
		pop, x_raw, sthresh_q=.1, run_dir=dirpath
	)

	img = visualization.visualize_tree(best_forest[best_scores.index(min(best_scores))], run_dir=dirpath)
	newforest , newscores = population.extract_n_best_trees(best_forest, best_scores, -1, run_dir=dirpath)

	x_ = transforms.forest2features(
		population=newforest,
		x_raw=x_raw
	)

	ynew = np.roll(x_raw[:, 3], shift=-1)
	y_ = np.log(ynew / x_raw[:, 3])

	X_train, X_test, y_train, y_test = train_test_split(x_, y_, test_size=0.3, shuffle=False)

	scaler = StandardScaler()
	X_train = scaler.fit_transform(X_train)
	X_test = scaler.transform(X_test)

	model, history = evaluation.standard_NN_construction(X_train, y_train)
	evaluation.standard_NN_evaluation(X_train, X_test, y_train, y_test, model, history, dirpath)
	return best_forest, best_scores


def random_sample_n_inverse_weighted(
	scores	:	list,
	n		:	int,
	let_dupe:	bool	=	True
):
	'''
	This function returns the indices of selected items.
	items are selected at random after assigning weights based on inverse scores.
	'''

	selected_indices = []

	raw_scores = scores.copy()
	raw_scores_rnd = [round(x, 8) for x in raw_scores]
	inv_scores = [1/x for x in raw_scores]
	
	wscore_vol = 0
	for invs in inv_scores:
		wscore_vol+=invs

	for iter in range(n):
		randval = np.random.uniform(low=0, high=wscore_vol)

		sorted_inv_scores = sorted(inv_scores, reverse=True)

		if(let_dupe == False):
			for i in range(iter):
				randval -= sorted_inv_scores[i]

			search_idx = iter
		else:
			search_idx = 0

		while(randval>sorted_inv_scores[search_idx]):

			randval -= sorted_inv_scores[search_idx]
			search_idx+=1
		
		selected_indices.append(
			raw_scores_rnd.index(
				round(1/sorted_inv_scores[search_idx], 8)
			)
		)

	if(len(selected_indices)!=n):
		raise ValueError(f"why does # selected indices not equal n?")
	
	return selected_indices

