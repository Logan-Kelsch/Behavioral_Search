
import random
import genetic_algorithm.evaluation as evaluation
import genetic_algorithm.transforms as transforms	
import genetic_algorithm.population as population
import genetic_algorithm.optimize as optimize
import genetic_algorithm.population as poppy
import genetic_algorithm.utility as utility
import genetic_algorithm.mutation as mutation
import genetic_algorithm.reproduction as reproduction
import genetic_algorithm.visualization as visualization
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import imageio
import io
import gc
import re
from pathlib import Path

def optimize_constants(
	population  :   list,
	x_raw		:	np.ndarray,
	sthresh_q	:	float	=	0.25,
	max_iter    :   int     =   -1,
	dyn_sthresh :   bool    =   True,
	run_dir		:	str		=	None
):
	assert run_dir!=None, 'Must assert a run directory to track progress.'
	
	loop_forest = population

	p_arr = transforms.forest2features(loop_forest, x_raw)

	p_scores, p_treelist, p_scorelist = evaluation.evaluate_forest_newer(
		p_arr, x_raw[:, 3], lag_range=(2, 4)
	)

	#print(p_scorelist)

	p_scorelist = utility.quickfix_score_to_loss(p_scorelist)

	#print(p_scorelist)

	#initialize score comparisons to track bests

	p_bests = p_scorelist.copy()
	p_bests_iter = [[] for _ in range(len(p_scorelist))]
	for s, score in enumerate(p_scorelist):
		p_bests_iter[s].append(score)

	#show current scoring of forest
	#plt.scatter(range(len(p_bests)), p_bests)
	#plt.title("bests")
	#plt.show()

	#takes the score marking the top sthersh
	sthresh = sorted(p_bests, reverse=False)[int(np.floor(len(p_bests)*sthresh_q))]
	#print(f"sthresh:{sthresh}")
	#print(f"Score extrema: [{min(p_bests)}, {max(p_bests)}]")

	norm_scores = np.asarray([sthresh/score for score in p_scorelist], dtype=np.float32)
	#plt.scatter(range(norm_scores.shape[0]), norm_scores)
	#plt.title("norm scores")
	#plt.show()

	#print(f'shape norm scores:{norm_scores.shape}')

	#for half life, = 0.693147
	kappa = 0.5#-np.log(0.5) 

	#this is also equal to (norm_score of sthresh) / (1 - e^-kappa)
	init_satiation = 2

	#intialize satiation
	satiation = []
	for i in range(norm_scores.shape[0]):
		satiation.append(init_satiation)

	#time roll satiation
	for i in range(len(satiation)):
		satiation[i] = satiation[i]*kappa + norm_scores[i]

	#print(f'shape satiation:{satiation.shape}')

	norm_satiation = np.asarray([s/init_satiation for s in satiation], dtype=np.float32)

	#plt.scatter(range(norm_satiation.shape[0]), norm_satiation)
	#plt.title("norm satiation")
	#plt.show()

	frames = []
	frames_ns = []
	frames_dfig = []
	frames_daxs = []

	frames.append(p_bests)
	frames_ns.append(norm_satiation)

	x_viz = pd.DataFrame(p_arr)
	d_fig, d_axes = visualization.visualize_all_distributions(x=x_viz, show=False)

	frames_dfig.append(d_fig)
	frames_daxs.append(d_axes)

	#print(f'shape normsatiation:{norm_satiation.shape}')

	desperation = np.asarray([(1/(ns**2)) for ns in norm_satiation])

	#print(f'shape desperation:{desperation.shape}')

	will_die = np.full((len(p_scorelist)), fill_value=1, dtype=np.int8)
	n_to_die = will_die.sum()

	stable_time = 0

	still_optimizing = bool(stable_time<2)

	iteration = 0

	#print('entering opt loop')

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

		x_viz = pd.DataFrame(forfeat_batches[0])
		d_fig, d_axes = visualization.visualize_all_distributions(x=x_viz, show=False)

		frames_dfig.append(d_fig)
		frames_daxs.append(d_axes)

		#print(f'opt-- forfeat[0] shape: {forfeat_batches[1].shape}')

		best_forest, best_scores = evaluation.get_best_forest(
			forfeat_batches=forfeat_batches,
			forest_batches=forest_batches,
			prll_idx_batches=prll_idx_batches,
			close_prices=x_raw[:, 3],
			lag_range=(2, 4)
		)

		#print(best_scores)

		best_scores = utility.quickfix_score_to_loss(best_scores)

		#print(best_scores)

		for i in range(len(p_bests_iter)):
			p_bests_iter[i].append(best_scores[i])

		for i in range(len(p_bests)):
			p_bests[i] = min(p_bests_iter[i])

		#show current scoring of forest
		#plt.scatter(range(len(p_bests)), p_bests)
		#plt.title("bests")
		#plt.show()

		frames.append(best_scores.copy())

		loop_forest = best_forest.copy()


		#takes the score marking the top sthersh
		sthresh = sorted(best_scores, reverse=False)[int(np.floor(len(best_scores)*sthresh_q))]
		#print(f"sthresh:{sthresh}")
		#print(f"Score extrema:[{min(p_bests)}, {max(p_bests)}]")

		norm_scores = np.asarray([sthresh/score for score in best_scores], dtype=np.float32)
		#plt.scatter(range(norm_scores.shape[0]), norm_scores)
		#plt.title("norm scores")
		#plt.show()

		#time roll satiation
		for i in range(len(satiation)):
			satiation[i] = satiation[i]*kappa + norm_scores[i]

		for i in range(len(satiation)):
			norm_satiation[i] = satiation[i]/init_satiation

		#plt.scatter(range(norm_satiation.shape[0]), norm_satiation)
		#plt.title("norm satiation")
		#plt.show()

		frames_ns.append(norm_satiation.copy())

		desperation = 1/(norm_satiation**2)

		death_mask = norm_satiation < 1
		will_die = death_mask.astype(int)

		if(bool(will_die.sum()<n_to_die)):
			stable_time = 0
		else:
			stable_time += 1

		still_optimizing = bool(stable_time<2)

		n_to_die = min(will_die.sum(), n_to_die)

		iteration+=1

		print(f"Expected to die: {will_die.sum()} @ Iteration #{iteration}")

	best_scores_over_time = []

	#print('opt loop completed')

	for j in range(len(p_bests_iter[0])):
		time_best = p_bests_iter[0][j]
		for i in range(1,len(p_bests_iter)):
			if(p_bests_iter[i][j]<time_best):
				time_best = p_bests_iter[i][j]
		best_scores_over_time.append(time_best)

	
	#complete norm satiation gif creation
	norms_frames_out = []
	ymin, ymax = 9999, -9999
	for frame in frames_ns:
		if(min(frame)<ymin):
			ymin = min(frame)*0.9
		if(max(frame)>ymax):
			ymax = max(frame)*1.1
	for i, frame in enumerate(frames_ns):
		fig, ax = plt.subplots()
		ax.set_ylim(ymin, ymax)
		ax.scatter(range(len(frame)), frame)
		ax.set_title(f"bests, iter: {i}")
		buf = io.BytesIO()
		fig.savefig(buf, format='png')
		plt.close(fig)
		buf.seek(0)
		norms_frames_out.append(imageio.imread(buf))

	#complete score gif creation
	score_frames_out = []
	ymin, ymax = 9999, -9999
	for frame in frames:
		if(min(frame)<ymin):
			ymin = min(frame)*0.9
		if(max(frame)>ymax):
			ymax = 1.05
	for i, frame in enumerate(frames):
		fig, ax = plt.subplots()
		ax.set_ylim(ymin, ymax)
		ax.scatter(range(len(frame)), frame)
		ax.set_title(f"bests, iter: {i}")
		buf = io.BytesIO()
		fig.savefig(buf, format='png')
		plt.close(fig)
		buf.seek(0)
		score_frames_out.append(imageio.imread(buf))

	#complete distribution gif creation
	distr_frames_out = []
	dist_frames = []
	for df in range(len(frames)):
		dist_frames.append((frames_dfig[df], frames_daxs[df]))
	for i, (fig, axes) in enumerate(dist_frames):
		axes[0].set_title(f"Distributions, iter: {i}")
		buf = io.BytesIO()
		fig.savefig(buf, format='png')
		plt.close(fig)
		buf.seek(0)
		distr_frames_out.append(imageio.imread(buf))

	#navigating runs folder
	
	best_scores_path = run_dir / 'best_scores.gif'
	normal_satn_path = run_dir / 'normal_satn.gif'
	featr_distr_path = run_dir / 'featr_distr.gif'

	#save gifs of progress
	imageio.mimsave(str(featr_distr_path), distr_frames_out, fps=3, loop=0)
	imageio.mimsave(str(best_scores_path), score_frames_out, fps=3, loop=0)
	imageio.mimsave(str(normal_satn_path), norms_frames_out, fps=3, loop=0)

	return loop_forest, p_bests, best_scores_over_time


def optimize_reproduction(
	init_size	:	int	=	250,
	init_dpth	:	int	=	5,
	step_size	:	float	=	0.1,
	iterations:	int	=	100,
	decay	:	float = 0.90,
	init_x	:	tuple	=	(0.5, 0.5)

):
	import tensorflow as tf
	
	data = pd.read_csv("../data/ES15.csv")
	x_raw = data.values

	x = np.array(init_x, dtype=float)

	dirpath = utility.fetch_new_run_dirpath()
	iterpath = dirpath / 'iter_0'
	iterpath.mkdir(exist_ok=True)


	#generate population, optimize
	best_forest = population.generate_random_forest(init_size, init_dpth)
	best_forest, best_scores, best_overtime = optimize.optimize_constants(
		best_forest, x_raw, sthresh_q=.15, run_dir=iterpath
	)
	#create starting point MERC reproduction, optimize
	best_forest = reproduction.reproduce(best_forest, best_scores, dflt_dpth=init_dpth, MERC=merc_from_2d(x))
	#optimize constants in forest
	best_forest, best_scores, best_overtime = optimize.optimize_constants(
		best_forest, x_raw, sthresh_q=.15, run_dir=iterpath
	)
	
	best_loss = loss_nn(best_forest, best_scores, x_raw, iterpath)
	print(f'init: {x}, {best_loss}')
	path = [x.copy()]

	iter = 1

	for iter in range(iterations):
		print(f'iter: {iter}')
		iterpath = dirpath / f'iter_{iter}'
		iterpath.mkdir(exist_ok=True)

		#random direction
		direction = np.random.randn(2)
		direction /= np.linalg.norm(direction)

		#propose new point
		candidate = x + step_size * direction
		candidate = np.clip(candidate, 0, 1)

		#generate
		best_forest = reproduction.reproduce(best_forest, best_scores, dflt_dpth=init_dpth, MERC=merc_from_2d(x))

		#optimize constants in forest
		best_forest, best_scores, best_overtime = optimize.optimize_constants(
			best_forest, x_raw, sthresh_q=.15, run_dir=iterpath
		)

		#evaluate
		loss_c = loss_nn(best_forest, best_scores, x_raw, iterpath)

		if(loss_c<best_loss):
			x = candidate
			best_loss = loss_c

		#decay step size
		step_size *= decay
		path.append(x.copy())
		print(f'cndt: {candidate}, {loss_c}')
		print(f'best: {x}, {best_loss}')


	return x, best_loss, np.array(path)


def merc_from_2d(
	point
):
	"""
	Given a point (x, y) in [0,1]x[0,1], returns
	areas of triangles from each side: [bottom, right, top, left].
	"""
	x, y = point
	sides = [
		((0, 0), (1, 0)),  # bottom
		((1, 0), (1, 1)),  # right
		((1, 1), (0, 1)),  # top
		((0, 1), (0, 0)),  # left
	]
	areas = []
	for (x0, y0), (x1, y1) in sides:
		cross = abs((x1 - x0)*(y - y0) - (y1 - y0)*(x - x0))
		areas.append(cross / 2)
	return tuple(areas)

def loss_nn(
	best_forest, best_scores, x_raw, dirpath
):
	import tensorflow as tf

	ynew = np.roll(x_raw[:, 3], shift=-1)
	y_ = np.log(ynew / x_raw[:, 3])

	#NN feature prep
	img = visualization.visualize_tree(best_forest[best_scores.index(min(best_scores))], run_dir=dirpath)
	newforest , newscores = population.extract_n_best_trees(best_forest, best_scores, 16, run_dir=dirpath)

	#turning forest into feature set
	x_ = transforms.forest2features(
		population=newforest,
		x_raw=x_raw
	)

	#NN data prep
	X_train, X_test, y_train, y_test = train_test_split(x_, y_, test_size=0.3, shuffle=True)
	scaler = StandardScaler()
	X_train = scaler.fit_transform(X_train)
	X_test = scaler.transform(X_test)

	#NN interpretation
	model, history = evaluation.standard_NN_construction(X_train, y_train, epochs=100)
	loss_NN = evaluation.standard_NN_evaluation(X_train, X_test, y_train, y_test, model, history, dirpath)

	#memory management for long term iterative looping
	tf.keras.backend.clear_session()
	del model, history
	gc.collect()

	return loss_NN