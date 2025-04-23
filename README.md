This project implements a part of speech tagger (POS) using a hidden markov model (HMM) and the Viterbi algorithm. The goal is to accurately assign syntactic categories (e.g., noun, verb, determiner) to each word in a sentence, even when words are ambiguous. An HMM is the statistical framework used to model sequences where the underlying states (POS in this project) are hidden and the observations (words in this project) are visible. I then wrote a Viterbi algorithm which, given an HMM and an observed sequence of words, computes the most likely sequence of hidden states, i.e. finding the best POS tag sequence for each sentence. 

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
