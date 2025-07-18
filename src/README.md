Source code folder


### New path of this project

#### background:
After watching the 'Thronglets' episode of 'Black Mirror', a thought was inspired in my head. The developer created a
genetic algorithm based game with a friendly population of a thing that was truly freely evolving within some bounds of transformations after observing utility/value. <br>
This was not exactly how I had structured this project, and the difference was in the depth of understanding. <br>
Today I had 5 hours in the car to brainstorm, and had realized exactly what limits of understanding my project was floating on. <br>
Although I had thought out a good half of it, I have learned to understand this:
- My featureset that may be generated will simply be an opportunity to generate genes with pre-structured logic.
- My featureset is nothing more than several layers of transformations to raw data for each feature (depth of only 1-4! wow!).
- I have determined which transformations are logically observable and logically stackable without redundancy or data drift.
I will move forward with this project where INITIAL development will PICK UP at gene generation from pre-made datapoints! <br>
This allows me to later remove my featureset and allow the genes to explore infinitely the raw data through transformation based evolution. <br>
I may also consider moving forward with simply not creating an initial featureset.

### General order of development (Revised: 6/8/2025)

- 1 collect data, multiple indices and time frames
- 2 create initial evo-based feature transformation functions
- 3 create universal clean-working-dataset-array functionality for clean data
- 4 create initial feature distribution observation/utility functions
- 5 create boolean-set gene structure and universal gene structure and serialization
- 6 create gene initialization functionality
- 7 create gene evaluation functionality
- 8 create intra-dimensional topology search functionality
- 9 create gene evolution functionality
- 10 consider some kind of evolutionary tree that tracks discovered utility that maybe optimizes gene evolution
- 11 expand evaluation of models for training models
- 12 develop universal observation node interpretation method (possibly already done)
- 13 implement NN into gene structures
- 14 implement HMM into gene structures
- 15 implement MDP into gene structures
- 16 visualization methods
- WIP



### INITIAL General order of development for this research

- 1 collect data, multiple indices and time frames
- 2 create initial set of features
- 3 create distribution detection and association functionality
- 4 create utility functionality for referencing feats (prior ex: fss)
- 5 create state generation algorithm from any feature set by considering distributions
- 6 create proof of concept for NN with initializable variables for random generation
- 7 create proof of concept for MDP with initializable variables for random generation
- 8 create proof of concept for HMM with initializable variables for random generation
- 9 Implement variable-node structure for pattern based gene (node)
- 10 Implement variable-node structure for NN/HMM/MDP full implementation for random generation
- 11 Implement capacity for node ('gene') to translate between node-type ('genome'),
  - 11.1 AKA. ensure universal congruency between random parameterization and feature perception, where unique node-based analytical approach is on the top of procedural stack.
- 12 implement gene + population serialization/deserialization
- 13 create custom population generation functions
- 14 explore segment entry exit theory
- 15 create differentiation in segment entry and exit
- 16 potentially explore segment optimization or solution space exploring
- 17 consider forbidden areas of solution space to be directly associated with exit conditioning
- 18 optimize all functionality
- 19 implement monte carlo and walk forward evaluation
- 20 create one click functions for visualization of results or dynamics of search
- 21 create results compilation functionality for easy visualization
- 22 execute large scale searches and compile findings

test test
