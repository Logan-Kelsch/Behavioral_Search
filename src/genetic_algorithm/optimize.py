
import random
import genetic_algorithm.evaluation as evaluation
import genetic_algorithm.transforms as transforms	
import genetic_algorithm.population as poppy
import genetic_algorithm.utility as utility
import genetic_algorithm.mutation as mutation
import numpy as np

def optimize_constants(
	population  :   list,
	x_raw		:	np.ndarray,
	sthresh_q	:	float	=	0.25,
	max_iter    :   int     =   -1,
	dyn_sthresh :   bool    =   True
):
	
	loop_forest = population

	p_arr = transforms.forest2features(loop_forest, x_raw)

	p_scores, p_treelist, p_scorelist = evaluation.evaluate_forest(
		p_arr, x_raw[:, 3], n_bins=300, lag_range=(2, 4)
	)

	#print(p_scorelist)

	p_scorelist = utility.quickfix_score_to_loss(p_scorelist)

	print(p_scorelist)

	#initialize score comparisons to track bests

	p_bests = p_scorelist.copy()
	p_bests_iter = [[] for _ in range(len(p_scorelist))]
	for s, score in enumerate(p_scorelist):
		p_bests_iter[s].append(score)

	#takes the score marking the top sthersh
	sthresh = sorted(p_bests, reverse=True)[int(np.floor(len(p_bests)*sthresh_q))]

	norm_scores = np.asarray([sthresh/score for score in p_scorelist], dtype=np.float32)

	#print(f'shape norm scores:{norm_scores.shape}')

	#for half life, = 0.693147
	kappa:np.float32 = -np.log(0.5) 

	#this is also equal to (norm_score of sthresh) / (1 - e^-kappa)
	init_satiation = 2

	satiation = np.full((norm_scores.shape[0]), fill_value=init_satiation)

	#print(f'shape satiation:{satiation.shape}')

	norm_satiation = np.asarray([init_satiation/s for s in satiation], dtype=np.float32)

	#print(f'shape normsatiation:{norm_satiation.shape}')

	desperation = np.asarray([(1/(ns**2)) for ns in norm_satiation])

	#print(f'shape desperation:{desperation.shape}')

	will_die = np.full((len(p_scorelist)), fill_value=1, dtype=np.int8)
	n_to_die = will_die.sum()

	stable_time = 0
	stable_time = (stable_time*int(will_die.sum()==n_to_die))+int(will_die.sum()==n_to_die) 

	still_optimizing = bool(stable_time<2)

	iteration = 0

	print('entering opt loop')

	while(still_optimizing):

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

			curr_oplist = oplists[oi].copy()

			#print(oi)

			#now go through all operations in the oplist to 
			#see what can be mutated		
			for i in range(len(curr_oplist)):

				#we are looking for operations with either
				#delta, delta2, const_alpha (flag: 1), kappa

				#print(len(curr_oplist))

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
						edit_oplist_over[i][2] = (over,)
						edit_oplist_under[i][2] = (under,)
						oplists.append(edit_oplist_under)
						orig_idx.append(orig_idx[oi])

					#for this case, we are going to check if 
					#flag is 1 (if alpha is a constant), then shift alpha if so
					case -5|-6:

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
						if(which_delta==0):
							edit_oplist_over[i][2] = (over, edit_oplist_over[i][2][1])
							edit_oplist_under[i][2] = (under, edit_oplist_over[i][2][1])
						else:
							edit_oplist_over[i][2] = (edit_oplist_over[i][2][0], over)
							edit_oplist_under[i][2] = (edit_oplist_over[i][2][0], under)
						oplists.append(edit_oplist_under)
						orig_idx.append(orig_idx[oi])

					#for this case, we are going to shift kappa
					case 8:

						edit_oplist_over = curr_oplist.copy()
						edit_oplist_under = curr_oplist.copy()
						under, over = mutation.mutate_constant(
							type='kappa', val=curr_oplist[i][2][0], dev=desperation[oi]
						)
						edit_oplist_over[i][2] = (over,)
						edit_oplist_under[i][2] = (under,)
						oplists.append(edit_oplist_under)
						orig_idx.append(orig_idx[oi])

		#now we have fully expanded oplists to all mutations and orig_idx to be parallel original tree indices
		#of loop forest for where it was mutated from.
		#we now also know that all indices of these prll lists >= orig_length are the mutations and can
		#be made into batches, scored, argmined performance, and replaced in the loopforest with best scores added to p_bests
		forest_batches, prll_idx_batches = poppy.oplist2forests(
			oplists=oplists,
			prll_idx=orig_idx,
			batch_size=orig_length
		)

		#print(len(forest_batches[1]))
		#print(transforms.get_oplist(forest_batches[1][0]))

		forfeat_batches = []

		for batch in range(len(forest_batches)):
			forfeat_batches.append(transforms.forest2features(forest_batches[batch], x_raw))

		#print(f'opt-- forfeat[0] shape: {forfeat_batches[1].shape}')

		best_forest, best_scores = evaluation.get_best_forest(
			forfeat_batches=forfeat_batches,
			forest_batches=forest_batches,
			prll_idx_batches=prll_idx_batches,
			close_prices=x_raw[:, 3],
			lag_range=(2, 4),
			n_bins=300
		)

		#print(best_scores)

		best_scores = utility.quickfix_score_to_loss(best_scores)

		print(best_scores)

		for i in range(len(p_bests_iter)):
			p_bests_iter[i].append(best_scores[i])

		for i in range(len(p_bests)):
			p_bests[i] = min(p_bests_iter[i])

		loop_forest = best_forest.copy()


		#takes the score marking the top sthersh
		sthresh = sorted(best_scores, reverse=True)[int(np.floor(len(best_scores)*sthresh_q))]

		

		norm_scores = np.asarray([sthresh/score for score in best_scores], dtype=np.float32)

		satiation = satiation*kappa + norm_scores

		norm_satiation = init_satiation/satiation

		desperation = 1/(norm_satiation**2)

		death_mask = norm_satiation < 1
		will_die = death_mask.astype(int)

		stable_time = (stable_time*int(bool(will_die.sum()==n_to_die)))+int(bool(will_die.sum()==n_to_die)) 

		still_optimizing = bool(stable_time>2)

		n_to_die = will_die.sum()

		iteration+=1
		print(f'ending opt #{iteration}')

	best_scores_over_time = []

	print('opt loop completed')

	for j in range(len(p_bests_iter[0])):
		time_best = p_bests_iter[0][j]
		for i in range(1,len(p_bests_iter)):
			if(p_bests_iter[i][j]<time_best):
				time_best = p_bests_iter[i][j]
		best_scores_over_time.append(time_best)

	return loop_forest, p_bests, best_scores_over_time