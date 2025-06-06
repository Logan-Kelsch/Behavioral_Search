'''
Dev: Logan Kelsch - 6/6/2025

This file will take some given csv file with created format 
and generate a FULL csv file of all desirable features.

(ex: 'candle' expecting):
high, low, close, volume, time

This file will consist of a main generation function,
as well as feature generation functions AND feature transformation functions.

Feature generation functions with be methods of pulling traditional time-series dimensionality from the data.
Feature transformation functions with be more universal methods of ranging or exciting time-series dimensionality from ANY time series data.

Current transforming function list:
- No transformation				(Identity,		ID, 0)
- Displacement from mean 		(demeaned, 		DM, 1)
- Standard deviation from mean 	(Volatility,	VL, 1)
- Min-Max Range Normalization 	(Stoch, 		MM, 3)
- Hawkes Self-Exciting Process 	(Hawkes,		HP, 2)

A direction graph for how these transformations can be applied on each other:

	_________>	DM ______\
   /			\/	     \/
ID __________>	MM	<____HP
   \			/\		 /\
	_________>	VL ______/
'''

