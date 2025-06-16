## PMF_ttree_generation_delta: ##

This image details the probability of some given value being selected for delta values in random generation of a transformation node. <br>
Reason for this distribution:
- More recent data is more likely to be useful when considering time axis.
- using inverse transform sampling, scaled the distribution out by factor of 26 (26 15mins in one market day).
- capping the distribution at 240 for minimizing dataset clipping and chance of unreasonable generations.

