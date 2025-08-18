import log as logs
import matplotlib.pyplot as plt
from typing import Literal
import random
import evaluation
import transforms
import serialization
import optimize
import population as poppy
import utility
import mutation
import reproduction
import visualization
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import copy
import numpy as np
import pandas as pd
import imageio
import io
import gc

def optimize_constants(
	population  :   list,
	x_raw		:	np.ndarray,
	sthresh_q	:	float	=	0.25,
	max_iter    :   int     =   -1,
	dyn_sthresh :   bool    =   True,
	run_dir		:	str		=	None,
	vizout		:	bool	=	True
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

		#print(f"Expected to die: {will_die.sum()} @ Iteration #{iteration}")

		if(max_iter>-1):
			if(iteration>=max_iter):
				break

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

	if(vizout):
		#save gifs of progress
		imageio.mimsave(str(featr_distr_path), distr_frames_out, fps=3, loop=0)
		imageio.mimsave(str(best_scores_path), score_frames_out, fps=3, loop=0)
		imageio.mimsave(str(normal_satn_path), norms_frames_out, fps=3, loop=0)

	del forfeat_batches, forest_batches
	del buf, fig, ax, axes, frames, frames_daxs, frames_dfig, frames_ns, dist_frames, distr_frames_out, norms_frames_out, score_frames_out
	
	gc.collect()

	return loop_forest, p_bests, best_scores_over_time

def optimize_keystone(
	init_size	:	int		=	100,
	init_dpth	:	int		=	3,
	iterations	:	int		=	500,
	survival	:	float	=	0.8,
	atr_thresh	:	float	=	1,
	plratio		:	float	=	1,
	dyn_theta	:	bool	=	True,
	dsprt_time	:	int		=	25,
	step_frac	:	float	=	0.37,

	step_size	:	float	=	0.05,
	init_lambda	:	tuple	=	(0.33, 0.33, 0.34),
	viz_mode	:	Literal['full','path']	=	'path',
	dcay_mode	:	Literal['decay','performance','pdecay']	=	'decay',
	decay_rate	:	float	=	0.996,
	loss_source	:	str	=	'best_invf1_balanced'
):
	np.seterr(all='ignore')
	logs.report_deep_globals()
	data = pd.read_csv("../../data/ES15.csv")
	x_raw = data.values.copy()
	del data

	if('balanced' in loss_source):
		use_bal_coef = True
	else:
		use_bal_coef = False

	ln_plratio = np.log(plratio)

	best_lambda = np.array(init_lambda, dtype=float)

	dirpath = utility.fetch_new_run_dirpath()

	match(viz_mode):
		case 'full':
			vizout=True
			iterpath = dirpath / 'iter_0'
			iterpath.mkdir(exist_ok=True)
		case 'path':
			vizout=False
			iterpath = dirpath

	#generate population, optimize
	p_forest = poppy.generate_random_forest(init_size, init_dpth)
	p_scores = [1]*len(p_forest)
	
	best_loss = 1
	loss_c = best_loss

	#sspace is solution space
	best_sspace_pos = [atr_thresh, ln_plratio, survival]
	best_sspace_pos = np.around(np.array(best_sspace_pos, dtype=float), 4).tolist()

	print(f'init: {best_sspace_pos}, {best_loss}')
	pscr = [best_loss]
	path = [best_sspace_pos.copy()]

	if(dcay_mode == 'decay'):
		step_size /= decay_rate

	alpha = 1
	#-1 signals to not use theta for any reason
	#per declaring dynamic theta False
	theta = -1
	stagnant = dsprt_time
	iter = 1

	for iter in range(iterations):
		print(f'iter: {iter}')
		match(viz_mode):
			case 'full':
				iterpath = dirpath / f'iter_{iter}'
				iterpath.mkdir(exist_ok=True)
			case 'path':
				#the path to file saving does not need to change
				pass

		#update step size
		if(dcay_mode == 'decay'):
			step_size *= decay_rate
		elif(dcay_mode == 'performance'):
			#step size here operates initially at 0.3 with domain of (0, 0.25)
			#and indef decay of size upon assumed continuous minima (local or global) finding
			step_size = ( (sorted(pscr).index(loss_c) + 1) / len(pscr) * 0.25)
		elif(dcay_mode == 'pdecay'):
			alpha *= decay_rate
			step_size = ( (sorted(pscr).index(loss_c) + 1) / len(pscr) * alpha)


		#we are going to offset theta if it is dynamically active.
		if(dyn_theta):
			theta = 1 - (max(stagnant,0)/dsprt_time)


		#cndt_lambda = propose_step_barycentric(best_lambda, step_size)
		#bringing in best_sspace_pos allows for continuation from best position
		atr_thresh, ln_plratio = propose_step_anytime2d(best_sspace_pos[0], best_sspace_pos[1], step_size, theta_offset=theta)

		cndt_sspace_pos = [atr_thresh, ln_plratio, survival]
		cndt_sspace_pos = np.around(np.array(cndt_sspace_pos, dtype=float), 4).tolist()

		best_forest = copy.deepcopy(p_forest)
		best_scores = copy.deepcopy(p_scores)

		#generate
		best_forest = reproduction.reproduce_scarce(best_forest, best_scores, size=init_size, sthresh=survival, dflt_dpth=init_dpth, MRC=best_lambda)

		x_ = transforms.forest2features(
			population=best_forest,
			x_raw=x_raw
		)

		sol_arr = evaluation.generate_solarr_atrplr(price_data=x_raw, atr_thresh=atr_thresh, ln_plratio=ln_plratio)


		best_scores = evaluation.evaluate_forest_atrplr(x_, solarr=sol_arr, bal_coef=use_bal_coef)
		
		#evaluate
		match(loss_source):
			case 'best_invf1':
				#domain: [0, 1]
				loss_c = min(round(1 - max(best_scores), 4), 1)

			case 'best_invf1_balanced':
				#domain: [0, 1]
				loss_c = min(round(1 - max(best_scores), 4), 1)

		
	

		# so I suppose here we will check to see if the loss_c is good enough
		# if the loss is enough we will propose a step forward in the atrthr/lnplr space

		# if the loss is not enough, we maybe can consider MERC optimization, but can implement later
		# otherwise we would just toss the cndt forest and try repr again on new cndt step

		# NOTE SUCCESS
		if(loss_c < survival):
			
			best_sspace_pos = cndt_sspace_pos
			best_loss = loss_c

			path.append(cndt_sspace_pos.copy())
			pscr.append(loss_c)

			stagnant = dsprt_time

			#print(f'TO COPY: {loss_fc(best_scores)}')

			p_forest = copy.deepcopy(best_forest)
			p_scores = copy.deepcopy(best_scores)
			pb = copy.deepcopy(best_loss)

			#step lower AFTER making step
			survival -= (survival-best_loss)*step_frac

		# NOTE FAILURE
		else:

			path.append(cndt_sspace_pos.copy())
			pscr.append(loss_c)
			path.append(best_sspace_pos.copy())
			pscr.append(best_loss)

			stagnant -= 1


		#to avoid extreme color values
		if(len(pscr)>=2):
			pscr[0] = pscr[1]

		if(loss_c<0 or best_loss<0):
			raise ValueError(f'NEGATIVE SCORE DETECTED. ENDING PROGRAM.'
							f'LOSS_FC generated from best_score === \n {best_scores}')
		
		print(f'cndt: {cndt_sspace_pos}, {loss_c}')
		print(f'crnt: {best_sspace_pos}, {best_loss}')
		

		if(iter%5==0):
			best_forest = poppy.remove_duplicates(best_forest)

		if(iter==iterations-1):
			
			logs.report_deep_locals()

		del best_forest, best_scores, x_

	gif_title = str(f"Best_L_{round(max(p_scores), 3)}")
	visualization.visualize_opt_path_3d(path, pscr, title=gif_title, dirpath=iterpath,
        frames=90, interval=150)
			
	best_idx = np.argmax(p_scores)

	img = visualization.visualize_tree(root=p_forest[best_idx], vizout=True, run_dir=iterpath)

	print(f'pb: {pb}')

	#print(loss_lm(best_forest=p_forest, best_scores=p_scores, x_raw=x_raw, dirpath=iterpath, vizout=vizout))

	#for i in range(len(p_forest)):
	#	print(transforms.get_oplist(p_forest[i]))

	serialization.save_forest(forest=p_forest, dirpath=iterpath, name='best.4st')
	serialization.save_deeplist(deep_list=[np.array(path), np.array(pscr)], dirpath=iterpath, name='path_pscr.hstry')

	return best_lambda, best_loss, np.array(path), np.array(pscr)	

def optimize_reproduction(
	init_size	:	int		=	50,
	init_dpth	:	int		=	3,
	step_size	:	float	=	0.25,
	epochs		:	int		=	50,
	pop_mode	:	Literal['new','rec']	=	'new',
	iterations	:	int		=	100,
	init_lambda	:	tuple	=	(0.25, 0.25, 0.25, 0.25),
	dcay_mode	:	Literal['decay','performance','pdecay']	=	'decay',
	step_mode	:	Literal['best','good','all']	=	'good',
	decay		:	float 	= 0.96,
	viz_mode	:	Literal['full','path']	=	'path',
	opt_const	:	bool	=	False,
	copt_iter	:	int		=	1,
	loss_source	:	Literal['nn','lm','fc']	=	'lm'
):
	np.seterr(all='ignore')

	logs.report_deep_globals()
	
	data = pd.read_csv("../../data/ES15.csv")

	x_raw = data.values.copy()

	del data

	best_lambda = np.array(init_lambda, dtype=float)

	dirpath = utility.fetch_new_run_dirpath()

	match(viz_mode):
		case 'full':
			vizout=True
			iterpath = dirpath / 'iter_0'
			iterpath.mkdir(exist_ok=True)
		case 'path':
			vizout=False
			iterpath = dirpath

	#generate population, optimize
	p_forest = poppy.generate_random_forest(init_size, init_dpth)
	p_scores = [1]*len(p_forest)
	
	'''match(loss_source):
		case 'nn':
			best_loss = loss_nn(best_forest=p_forest, best_scores=best_scores, epochs=epochs, x_raw=x_raw, dirpath=iterpath, vizout=vizout)
		case 'lm':
			best_loss = loss_lm(best_forest=p_forest, best_scores=best_scores, x_raw=x_raw, dirpath=iterpath, vizout=vizout)
		case 'fc':
			best_loss = loss_fc(best_scores)'''
	
	best_loss = 1
	loss_c = best_loss

	print(f'init: {best_lambda}, {best_loss}')
	pscr = [best_loss]
	path = [best_lambda.copy()]

	if(dcay_mode == 'decay'):
		step_size /= decay

	alpha = 1

	iter = 1

	for iter in range(iterations):
		print(f'iter: {iter}')
		match(viz_mode):
			case 'full':
				iterpath = dirpath / f'iter_{iter}'
				iterpath.mkdir(exist_ok=True)
			case 'path':
				#the path to file saving does not need to change
				pass

		#update step size
		if(dcay_mode == 'decay'):
			step_size *= decay
		elif(dcay_mode == 'performance'):
			#step size here operates initially at 0.3 with domain of (0, 0.25)
			#and indef decay of size upon assumed continuous minima (local or global) finding
			step_size = ( (sorted(pscr).index(loss_c) + 1) / len(pscr) * 0.25)
		elif(dcay_mode == 'pdecay'):
			alpha *= decay
			step_size = ( (sorted(pscr).index(loss_c) + 1) / len(pscr) * alpha)


		cndt_lambda = propose_step_barycentric(best_lambda, step_size)


		best_forest = copy.deepcopy(p_forest)
		best_scores = copy.deepcopy(p_scores)

		if(pop_mode=='new'):
			#generate population, optimize
			best_forest = poppy.generate_random_forest(init_size, init_dpth)
			
			if(opt_const):
				best_forest, best_scores, best_overtime = optimize.optimize_constants(
					best_forest, x_raw, sthresh_q=.15, run_dir=iterpath, vizout=vizout, max_iter=copt_iter
				)
			else:
				x_ = transforms.forest2features(
					population=best_forest,
					x_raw=x_raw
				)
				_, __, best_scores = evaluation.evaluate_forest_newer(x_, close_prices=x_raw[:, 3], lag_range=(1, 3))
				del _, __

		#generate
		best_forest = reproduction.reproduce(best_forest, best_scores, dflt_dpth=init_dpth, MERC=cndt_lambda)

		if(opt_const):
			#optimize constants in forest
			best_forest, best_scores, best_overtime = optimize.optimize_constants(
				best_forest, x_raw, sthresh_q=.15, run_dir=iterpath, vizout=vizout, max_iter=copt_iter
			)
		else:
			x_ = transforms.forest2features(
				population=best_forest,
				x_raw=x_raw
			)
			_, __, best_scores = evaluation.evaluate_forest_newer(x_, close_prices=x_raw[:, 3], lag_range=(1, 3))
			del _, __

		#evaluate
		match(loss_source):
			case 'nn':
				loss_c = loss_nn(best_forest=best_forest, best_scores=best_scores, epochs=epochs, x_raw=x_raw, dirpath=iterpath, vizout=vizout)
			case 'lm':
				loss_c = loss_lm(best_forest=best_forest, best_scores=best_scores, x_raw=x_raw, dirpath=iterpath, vizout=vizout)
			case 'fc':
				loss_c = loss_fc(best_scores)
		
		#print(f'LOSS_C: {loss_c, loss_fc(best_scores)}')
	
		if(step_mode == 'best'):

			# NOTE SUCCESS
			if(loss_c<best_loss):
				best_lambda = cndt_lambda
				best_loss = loss_c

				path.append(cndt_lambda.copy())
				pscr.append(loss_c)

				#print('new best.')

				#print(f'TO COPY: {loss_fc(best_scores)}')

				p_forest = copy.deepcopy(best_forest)
				p_scores = copy.deepcopy(best_scores)
				pb = copy.deepcopy(best_loss)

				#print(f'DD COPY: {loss_fc(p_scores)}')

			else:

				path.append(cndt_lambda.copy())
				pscr.append(loss_c)
				path.append(best_lambda.copy())
				pscr.append(best_loss)

		elif(step_mode == 'good'):

			#for a step to be taken under 'good' step mode
			#requires that the candidate position scores with in the top
			#quarter of all scores collected this far
			# NOTE SUCCESS

			if(loss_c <= ( sorted(pscr)[int(len(pscr)/20)] )):
				best_lambda = cndt_lambda
				best_loss = loss_c

				path.append(cndt_lambda.copy())
				pscr.append(loss_c)

				#print('new best.')

				p_forest = copy.deepcopy(best_forest)
				p_scores = copy.deepcopy(best_scores)
				pb = copy.deepcopy(best_loss)

			else:

				path.append(cndt_lambda.copy())
				pscr.append(loss_c)
				path.append(best_lambda.copy())
				pscr.append(best_loss)

		elif(step_mode == 'all'):

			# NOTE SUCCESS

			best_lambda = cndt_lambda
			best_loss = loss_c
		
			path.append(cndt_lambda.copy())
			pscr.append(loss_c)

			p_forest = copy.deepcopy(best_forest)
			p_scores = copy.deepcopy(best_scores)
			pb = copy.deepcopy(best_loss)

		#to avoid extreme color values
		if(len(pscr)==2):
			pscr[0] = pscr[1]

		if(loss_c<0 or best_loss<0):
			raise ValueError(f'NEGATIVE SCORE DETECTED. ENDING PROGRAM.'
							f'LOSS_FC generated from best_score === \n {best_scores}')
		
		print(f'cndt: {cndt_lambda}, {loss_c}')
		print(f'crnt: {best_lambda}, {best_loss}')

		if(iter==iterations-1 or iter%100==99):
			visualization.animate_opt_path_bary(np.array(path), np.array(pscr), title=f'i{iter}_path.gif', dir_path=iterpath)

			logs.report_deep_locals()

		del best_forest, best_scores, x_

	print(f'pb: {pb}')

	#print(loss_lm(best_forest=p_forest, best_scores=p_scores, x_raw=x_raw, dirpath=iterpath, vizout=vizout))

	#for i in range(len(p_forest)):
	#	print(transforms.get_oplist(p_forest[i]))

	serialization.save_forest(forest=p_forest, dirpath=iterpath, name='best.4st')
	serialization.save_deeplist(deep_list=[np.array(path), np.array(pscr)], dirpath=iterpath, name='path_pscr.hstry')

	return best_lambda, best_loss, np.array(path), np.array(pscr)


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

import math

def propose_step_anytime2d(val_1, val_2, step_size, theta_offset:float=-1, rot_atr:bool=False):
	"""
	Return (new_free, new_pos) such that
	  sqrt((new_free-val_free)^2 + (new_pos-val_pos)^2) == step_size,
	with new_pos >= val_pos (i.e. dpos in [0,step_size]) and
	new_free unconstrained (dfree in [-step_size, +step_size]).

	This samples uniformly over the semicircle of radius step_size
	where dpos ≥ 0.
	"""

	if(theta_offset == -1):
		#this case means we are NOT using a theta offset
		# pick an angle theta in [0, pi] uniformly
		th = random.random() * math.pi
		#throwing a chance of negative values in there in one dimention
		#net direction is still >=0 for all random.random().. I think????
		dfree = step_size * math.cos(th - (math.pi/4))
		dpos  = step_size * math.sin(th - (math.pi/4))

		return val_1 + dfree, val_2 + dpos
	
	else:
		#theta offset will be coming in as 0 being only positive pick
		#theta offset will be coming in as 1 being any direction pick

		#we are using sin/cos curves of (x - 3/4*pi)

		#theta coming in [0 , 1] scales uniform selection range into [pi/2 , 2pi]
		theta_range = ( theta_offset * 3 * math.pi + math.pi ) / 2
		
		#we will manage this around zero for simplicity
		#this being centered around zero is then moved to pi
		th = (random.random() - 0.5) * theta_range + math.pi
		

		curve_offset = 3 * math.pi / 4

		if(rot_atr==False):
			dcos = step_size * math.cos(random.random() * math.pi) / 2
		else:
			dcos = step_size * math.cos(th - curve_offset)
		dsin = step_size * math.sin(th - curve_offset)

		return val_1 + dcos, val_2 + dsin
		


def propose_step_barycentric(lambda_current, step_size):#
	"""
	Given current barycentric coords (shape (4,), sum to 1),
	propose a random step within the simplex:
	  - direction sampled in the hyperplane sum=0
	  - clipped to [0,1], then renormalized to sum=1
	"""
	# sample a random direction in R^4
	d = np.random.randn(4)
	# project onto hyperplane sum(d)=0
	d = d - d.mean()
	# normalize
	d /= np.linalg.norm(d)
	
	# propose new barycentric coordinate
	lambda_candidate = lambda_current + step_size * d
	# ensure non-negative
	lambda_candidate = np.clip(lambda_candidate, 0, None)
	# renormalize to sum=1
	lambda_candidate /= lambda_candidate.sum()
	return lambda_candidate


def barycentric_from_point(point):
	"""
	Given a point (x, y, z) inside the unit tetrahedron
	defined by vertices v0=(0,0,0), v1=(1,0,0), v2=(0,1,0), v3=(0,0,1),
	returns the normalized sub-volumes (barycentric coordinates)
	corresponding to each vertex. The four returned values sum to 1.
	
	Order of returns: (λ0, λ1, λ2, λ3), where λi is the fraction of the
	volume of the tetrahedron opposite vertex vi.
	"""
	x, y, z = point
	# vertices
	v0 = np.array([0.0, 0.0, 0.0])
	v1 = np.array([1.0, 0.0, 0.0])
	v2 = np.array([0.0, 1.0, 0.0])
	v3 = np.array([0.0, 0.0, 1.0])

	def tetra_volume(a, b, c, d):
		# Signed volume of tetrahedron (absolute)
		return abs(np.dot(b - a, np.cross(c - a, d - a))) / 6.0

	full_vol = tetra_volume(v0, v1, v2, v3)

	faces = [
		(v1, v2, v3),  # volume opposite v0
		(v0, v2, v3),  # opposite v1
		(v0, v1, v3),  # opposite v2
		(v0, v1, v2),  # opposite v3
	]

	p = np.array([x, y, z])
	sub_vols = [tetra_volume(p, a, b, c) for (a, b, c) in faces]
	# normalize so sum == 1
	bary_coords = [sv / full_vol for sv in sub_vols]
	return tuple(bary_coords)

def loss_fc(
	scores
):
	
	best_scores = sorted(scores, reverse=True)

	to_avg = int(np.ceil(len(best_scores)/4))

	avg = 0
	
	for i in range(to_avg):
		avg += (1 - best_scores[i])**2

	avg /= to_avg

	return avg

def loss_lm(
	best_forest, best_scores, x_raw, dirpath, vizout
):
	from sklearn.linear_model import LinearRegression

	model = LinearRegression(n_jobs=-1)

	ynew = np.roll(x_raw[:, 3], shift=-1)
	y_ = np.log(ynew / x_raw[:, 3])

	#newforest , newscores = population.extract_n_best_trees(best_forest, best_scores, 32, run_dir=dirpath, vizout=vizout)

	#turning forest into feature set
	x_ = transforms.forest2features(
		population=best_forest,
		x_raw=x_raw
	)

	#data prep
	X_train, X_test, y_train, y_test = train_test_split(x_, y_, test_size=0.3, shuffle=True, random_state=0)
	scaler = StandardScaler()
	X_train = scaler.fit_transform(X_train)
	X_test = scaler.transform(X_test)

	model.fit(X_train, y_train)
	y_pred = model.predict(X_test)
	y_pred_train = model.predict(X_train)

	self_r2, self_qacc = visualization.visualize_regression_eval(y_test=y_train, y_pred=y_pred_train, title='Self Test', run_dir=dirpath)
	ind_r2, ind_qacc = visualization.visualize_regression_eval(y_test=y_test, y_pred=y_pred, title='Independent Test', run_dir=dirpath)

	self_qacc = (self_qacc * 2 - 1)
	ind_qacc = (ind_qacc * 2 - 1)

	loss_LM = (
		(1-self_r2) * (1-self_qacc) * min(1-ind_r2**2, 1) * (1-ind_qacc**2)
	)

	del model

	return loss_LM


def loss_nn(
	best_forest, best_scores, epochs, x_raw, dirpath, vizout
):
	import tensorflow as tf

	ynew = np.roll(x_raw[:, 3], shift=-1)
	y_ = np.log(ynew / x_raw[:, 3])

	#NN feature prep
	img = visualization.visualize_tree(best_forest[best_scores.index(min(best_scores))], run_dir=dirpath, vizout=vizout)
	newforest , newscores = poppy.extract_n_best_trees(best_forest, best_scores, 16, run_dir=dirpath, vizout=vizout)

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
	model, history = evaluation.standard_NN_construction(X_train, y_train, epochs=epochs)
	loss_NN = evaluation.standard_NN_evaluation(X_train, X_test, y_train, y_test, model, history, dirpath, vizout=vizout)

	#memory management for long term iterative looping
	tf.keras.backend.clear_session()
	del model, history
	del X_train, X_test, y_train, y_test, x_
	gc.collect()

	return loss_NN

if __name__ == "__main__":
	print("running...")
	#optimize_reproduction()

	for i in range(8):
		optimize_keystone(init_size=200, iterations=50, atr_thresh=1.5, plratio=2.5, init_dpth=5, init_lambda=(0.0,0.25,0.75),step_frac=round((i+1)/25, 3))
	for i in range(8):
		optimize_keystone(init_size=200, iterations=50, atr_thresh=1.5, plratio=4, init_dpth=5, init_lambda=(0.0,0.25,0.75),step_frac=round((i+1)/25, 3))
	for i in range(8):
		optimize_keystone(init_size=200, iterations=50, atr_thresh=1.5, plratio=8, init_dpth=5, init_lambda=(0.0,0.25,0.75),step_frac=round((i+1)/25, 3))