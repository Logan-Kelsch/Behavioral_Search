Logan Kelsch - 5/27/25 <br>

After significant brainstorming and consideration, I have determined the general structure of my automated genetic algorithm.

## AUTOMATED GENETIC ALGORITHM CONSIDERATIONS: ##

I have determined that reproduction in prior versions is not as efficient as it should be for this kind of search.<br>
In this project we are working with an extensive number of dimensions that contain LIMITED SPACE.<br>
Because of this dimension based search, reproduction should be considered in these manners:

- Gene Reproduction should be fully-inclusive and a byproduct of 'surviving' in some dimension-local loss topology. 
- - INSTEAD of being randomly/partially-inclusive of genes within a defined survival threshold. 



## AUTOMATED GENETIC ALGORITHM PROCEDURE: ##

- A set of n (user-defined) random features within F (the provided set of featuresets) are selected for gene creation,
- - where each gene considers f (user-defined) number of features.
- All possible combinations, c, of f number of feature considerations are generated as genes.
- - Each gene specific set of considerable features call be called a dimension (of data), making this initial population "inter-dimensional". 
- Each gene will tread <= a fixed set of steps searching for more optimal loss.
- Each gene is provided a "satiation" attribute that follows the gene through intra-dimension loss topology treading.
- This satiation value will use Hawkes Process and an 'observed-loss' transformation function to excite in regards to a survival threshold. 
- - The duration of the search will be determined by if said gene's satiation does not excite above the required threshold.
- - The length of each tread will be based on the 'despiration' (inverse satiation) of the gene at that given time-step.
- - - This ensures local-minima avoidance if the minima is not sufficient. 
- - - The tread-size (learning-rate) will be random-uniform <= +/-despiration transformed to size of dimension
- Once all genes have reached some minima or died (exploding: despiration, max-tread), inter-dim loss is visualized, and intra-dim treading is mapped visually.
- Dead members of population are removed. Most recent observed loss for each gene is recorded.
- Surviving threshold is now set to top 50% of population if more than 50% survived, or is set to worst surviving gene. Surviving population is stored in long-term memory variable.
- Reproduction will happen at this stage and should clearly be chosen between INTRA and INTER dimensional, as one moves into more complex observable sets of features.
- INTRA-dimensional reproduction here is best:
- - A new population is made with random mutations across feature-space WITHIN the local featureSET for ANY (>=1) feature observed by some gene, for ALL surviving genes.
- - This fully new population will go through the exact same topology treading procedure defined above.
- - Surviving population of INTRA-dim offspring is concattenated and sorted into parent population, and same survival threshold applies, weeding out weakest genes.
- - INTRA phases of reproduction can be limited to a maximum survival threshold before either:
- - - quitting and saving all performance
- - - or before transitioning reproductive step into INTER-dimensional
- INTER-dimensional reproduction here is possible:
- - The caveat here is that all gene consideration sets of size f have been explored. 
- - INTER-dim reproduction should only be explored when user is okay with a search with increasing complexity, possibly memory risks during computation.
- - f can be limited here to bring a conclusion to the automated search.
- 
- Once some desirable gene(s) have been defined and saved for the long term, a pearson's r^2 correlation can be incorporated into the loss topology.
- - This will steer new genes away from converging to already discovered minima, diversifying findings.
- Once the AGA (Automaged-Genetic-Algorithm) has generated a set of genes, further performance evaluations can be executed:
- - Monte-Carlo Permutation Test
- - Walk-forward Evaluation