'''
Logan Kelsch - 6/8/2025

This file contains the functions that will be used for the manipulation of features
during the development / evolution stage of a gene or population.

The four TYPES of transformations (Phi):
- Phi-F (F):	Results in Stochastic Motion (Free Distribution)
- Phi-O (O):	Results in Oscillating Motion (About 0 or About 1)
- Phi-R (R):	Results in Ranging Motion (Within Fixed Value Range)
- Phi-D (D):	Results in Exciting or Decaying Motion (Hawkes Process)

All transformations (as of 6/8/2025, day theorized):
- Difference	(Subtraction)	{Phi-O1}
- Scale			(Division)		{Phi-O2}
- Normalized Ranging			{Phi-O3}
- MinMax Norm	(Ranging)		{Phi-R}
- Minimum, Maximum				{Phi-F1,F2}
- Average, Absolute				{Phi-F3,F4}
- Hawkes Process (Exciting)		{Phi-D2}

These transformations are stackable in any manner, and a final observable value after transformations can be represented by a tree.
Some transformations are redundant, and some obvservable transformations are subject to data drift, these will be hardcoded out of possibility.

Therefore, I have defined a state based system, on what can stack on what, and what <<is>> a final state and what is <not>.

	<F>		->	{ <F>	, <<O>>	, <<R>> }
    <<O>>	->	{ <<O>>	, <<D>>	, <<R>> }
    <<D>>	->	{ <<D>>	, <<R>> }
    <<R>>	->	{ <<R>> }
    
Variables Derived from each Transforming function:

This is being initially described in the Research Extentions latex file
'''