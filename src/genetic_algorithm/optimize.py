
import genetic_algorithm.evaluation as evaluation
import genetic_algorithm.transforms as transforms	
import numpy as np

def optimize_constants(
	population  :   list,
	x_raw		:	np.ndarray,
	sthresh		:	float	=	0.25,
	max_iter    :   int     =   -1,
	dyn_sthresh :   bool    =   True
):
	
	loop_forest = population

	p_arr = transforms.forest2features(loop_forest, x_raw)

	i = 0

	p_scores, p_treelist, p_scorelist = evaluation.evaluate_forest(
		p_arr, x_raw[:, 3], n_bins=300, lag_range=(2, 4)
	)

	#initialize score comparisons to track bests

	p_bests = np.array(len(p_scorelist), dtype=list)
	for s, score in enumerate(p_scorelist):
		p_bests[s].append(score)

	sthresh = sorted(p_scorelist)[int(np.floor(len(p_scorelist)*sthresh))]

	norm_scores = np.asarray([sthresh/score for score in p_scorelist], dtype=np.float32)

	#for half life, = 0.693147
	kappa:np.float32 = -np.log(0.5) 

	#this is also equal to (norm_score of sthresh) / (1 - e^-kappa)
	init_satiation = 2

	satiation = np.full((norm_scores.shape[0]), fill_value=init_satiation)

	norm_satiation = np.asarray([init_satiation/s for s in satiation], dtype=np.float32)

	desperation = np.asarray([(1/(ns**2)) for ns in norm_satiation])

	#get oplists
	oplists = [transforms.get_oplist(tree) for tree in loop_forest]

	#a parallel list of indices that follows 
	#where the new trees are generating from
	#to ensure proper overfitting avoidance and
	#ease of tracking optimization
	#this is initialized as an index list parallel to the original forest
	orig_idx = [i for i in range(len(loop_forest))]

	#go through all mutation possibilities and append to both lists
	pre_length = len(oplists)
	for oi in range(pre_length):

		curr_oplist = oplists[oi]
		edit_oplist = oplists[oi].copy()

		#iterate through oplist and make all
		#possible random generations for each original oplist

	'''
	So it looks like I can make two parallel giant lists
	of mutated oplists prllto original indices

	then I can break this into batches
	then I can turn the oplists batches into forests

	then I can evaluate all forests,
	replace all loop_forest trees with best tree for each orig_idx

	then I can evaluate the finalized forest
	and append new scores to p_bests
	'''

	#via trees from oplists
	
	#run sets and append scores to idx
	
	#condense best scoree point to new best
	
	#evaluate loop_forest
	
	#update sthresh
	
	#update norm scores
	
	#age satiation