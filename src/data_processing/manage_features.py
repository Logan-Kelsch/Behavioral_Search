'''
Ultimately what I would like .... wip




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