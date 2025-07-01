
import random
import genetic_algorithm.evaluation as evaluation
import genetic_algorithm.transforms as transforms	
import genetic_algorithm.mutation as mutation
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
	orig_length = len(oplists)

	for oi in range(orig_length):

		curr_oplist = oplists[oi]

		#now go through all operations in the oplist to 
		#see what can be mutated		
		for i in range(len(curr_oplist)):

			#we are looking for operations with either
			#delta, delta2, const_alpha (flag: 1), kappa

			match(curr_oplist[i][0]):

				#for these cases, we are either on a leaf
				#and I am not currently implementing raw data shifting
				#or we are flipping the sign.
				#both of these cases have no treadable dimensionality
				case 0|4:
					#go to next operation
					continue

				#for this case, we are going to shift delta
				case 1|2|3:

					edit_oplist_over = curr_oplist.copy()
					edit_oplist_under = curr_oplist.copy()
					under, over = mutation.mutate_constant(
						type='delta', val=curr_oplist[i][2][0], dev=desperation[oi]
					)
					edit_oplist_over[i][2][0] = over
					edit_oplist_under[i][2][0] = under
					oplists.append(edit_oplist_under)
					orig_idx.append(orig_idx[oi])

				#for this case, we are going to check if 
				#flag is 1 (if alpha is a constant), then shift alpha if so
				case 5|6:

					#currently, we are not going to mutate alpha,
					#for non-redundant value generation,
					#we would need to take in the distribution information
					#that is generated. this is not too silly,
					#but would take a lot of implementation,
					#therefore this can be done at a later time/date
					continue
					

				#for this case we are going to shift either
				#delta or delta2 based on coin flip.
				#tracking all paths for this manipulation would
				#take more implementation than what it is worth
				case 7:

					which_delta = random.randint(0,1)

					edit_oplist_over = curr_oplist.copy()
					edit_oplist_under = curr_oplist.copy()
					under, over = mutation.mutate_constant(
						type='delta', val=curr_oplist[i][2][which_delta], dev=desperation[oi]
					)
					edit_oplist_over[i][2][which_delta] = over
					edit_oplist_under[i][2][which_delta] = under
					oplists.append(edit_oplist_under)
					orig_idx.append(orig_idx[oi])

				#for this case, we are going to shift kappa
				case 8:

					edit_oplist_over = curr_oplist.copy()
					edit_oplist_under = curr_oplist.copy()
					under, over = mutation.mutate_constant(
						type='kappa', val=curr_oplist[i][2][0], dev=desperation[oi]
					)
					edit_oplist_over[i][2][0] = over
					edit_oplist_under[i][2][0] = under
					oplists.append(edit_oplist_under)
					orig_idx.append(orig_idx[oi])

	#now we have fully expanded oplists to all mutations and orig_idx to be parallel original tree indices
	#of loop forest for where it was mutated from.
	#we now also know that all indices of these prll lists >= orig_length are the mutations and can
	#be made into batches, scored, argmined performance, and replaced in the loopforest with best scores added to p_bests


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