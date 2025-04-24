Hidden Markov Models (HMMs) have proven an ubiquitous way to approach pattern recognition problems that enable us to infer something we cannot directly observe, even in the presence of noisy data. This type of model is driven by transitions from state to state, with the key assumption that the likelihood of moving to a future state depends only on the current state, not the entire history. The “Markov” property in its names refers to a mathematical stochastic process in which the probability of each event depends only on the state attained in the previous one. An HMM is defined by a set of hidden states, a set of observations, and a collection of weights (I.e. probabilities) which then determine their transition probabilities and emission probabilities In this project, the hidden states of part of speech (POS) tags, the observations are the words from an input of text, meaning a transition probabilitiy is the likelihood of moving from one POS to another POS (e.g. verb —> noun), and an emssion probabilities is likelihood of a POS producing a given word (e.g. noun —> “fish”). The weights are learned from training data and typically converted to log probabilities for computation. Once an HMM is trained, an algorithm for decoding can then be used to determine the best path through a given graph from data. In this project, the Viterbi algorithm is used to efficiently compute the most likely sequence of hidden states that could have produced a given sequence of observations. 

Code and output files are contained in HMMPartOfSpeechTagging, which includes:

HMM.java
- implements the core HMM logic, including training on labeled sentence/tag pairs and computing transition and emission probabilities.
- KFoldHMM.java contains an additional method to allow for k-fold cross validation testing 

Viterbi.java
- implementation of the Viterbi algorithm to decode the most likely tag sequence for a given input sentence
- handles backpointer tracking and final sequence decoding

HMMOutput
- stores the tagging results for a given test set
- KFoldOutput contains tagging results using k-folds cross validation
