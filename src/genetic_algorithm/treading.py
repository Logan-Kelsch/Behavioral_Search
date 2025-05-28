


def pop_topology_tread():
	return


def pop_satiation_timestep(
		
):
	'''
	### INFO: ###
	Satiation will be the attribute dictating behavior of a gene in the loss topology environment. <br>
	Satiation will be a self-exciting measure of observed loss by the gene over timesteps in the environment. <br>
	Satiation will use Hawkes Process (kernel) X_(i) = (e^-Kappa) * X_(i-1) + Theta_(i) <br>
	Since we have a survival threshold, we can define that threshold value (Theta) as the point of stability, allowing us to define our constant. <br>
	NOTE also that Theta will be a normalized loss value by function Phi, where survival threshold is brought to a universal value of 1 <br>
	NOTE that for this formula, stability over time exists (experiencing minimum acceptable loss perceived as (1)) under the following constrains:  <br>
	- In terms of X, => X =  1 / ( 1 - e^-K )
	- in terms of K, => K = ln( n / ( n-1 ) )
	### OVERVIEW: ###
	Satiation will be calculated with following points of significance:
	- Satiation = Decay_Constant * Last_Satiation + Just_Consumed
	- X_(i)     = (e^-Kappa)     * X_(i-1)        + Theta_(i)
	- Observed loss = L
	- Survival (Desirable loss) threshold = S
	- Decay_Constant = e^-Kappa
	- Phi(L) = S/L
	- Satiation = e^-ln(S) * Satiation[1] + 
	NOTE NOTE NOTE NEED TO DEFINE A GOOD KAPPA FOR SATIATION USING FORMULAS DERIVED!!!
	'''
	return

def get_desperation():
	return