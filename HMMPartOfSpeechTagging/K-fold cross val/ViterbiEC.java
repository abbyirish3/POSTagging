import java.util.*;

/**
 * Viterbi algorithm for HMM
 *
 * @author Abigail Irish
 */
public class ViterbiEC {
    String startState = "#";
    double unseenScore = -10;
    String bestState = null;
    double bestScore = Double.NEGATIVE_INFINITY;
    LinkedList<String> bestPath;

    public List<String> decodeEC(String[] words, HMMEC markov) {
        // convert sentence to lowercase
        for (int i = 0; i < words.length; i++) {
            words[i] = words[i].toLowerCase();
        }

        // list of maps for each observation, mapping it to its previous state
        List<Map<String, String>> backpointers = new ArrayList<>();

        //    currStates = { start }
        //    currScores = map { start=0 }
        Map<String, Double> currScores = new HashMap<>();
        currScores.put(startState, 0.0);

        // for i from 0 to # observations - 1
        for (int i = 0; i < words.length; i++) {
            String word = words[i];

            // nextStates = {}
            // nextScores = empty map
            Map<String, Double> nextScores = new HashMap<>();
            Map<String, String> currBackpointer = new HashMap<>();

            // for each currState in currStates
            for (String currState : currScores.keySet()) {
                double currScore = currScores.get(currState);
                Map<String, Double> transitions = markov.getTransitionProbs().get(currState);
                if (transitions == null) continue;

                // for each transition for the currState -> nextState
                for (Map.Entry<String, Double> entry : transitions.entrySet()) {
                    String nextState = entry.getKey();
                    double transitionScore = entry.getValue();

                    //    add nextState to nextStates
                    // nextScore = currScores[currState] +                       // path to here
                    // transitionScore(currState -> nextState) +     // take a step to there
                    // observationScore(observations[i] in nextState) // make the observation there
                    double obsScore = unseenScore;
                    Map<String, Double> obsMap = markov.getObservationProbs().get(nextState);
                    if (obsMap != null && obsMap.containsKey(word)) {
                        obsScore = obsMap.get(word);
                    }
                    double nextScore = currScore + transitionScore + obsScore;

                    // if nextState isn't in nextScores or nextScore > nextScores[nextState]
                    //set nextScores[nextState] to nextScore
                    if (!nextScores.containsKey(nextState) || nextScore > nextScores.get(nextState)) {
                        nextScores.put(nextState, nextScore);
                        currBackpointer.put(nextState, currState);
                    }
                }
            }
            // remember that pred of nextState @ i is curr
            // currStates = nextStates
            // currScores = nextScores
            backpointers.add(currBackpointer);
            currScores = nextScores;
        }

        for (Map.Entry<String, Double> entry : currScores.entrySet()) {
            if (entry.getValue() > bestScore) {
                bestScore = entry.getValue();
                bestState = entry.getKey();
            }
        }
        if (bestState == null) {
            System.err.println("Error: No valid path found.");
            return new ArrayList<>();
        }
        bestPath = new LinkedList<>();
        String state = bestState;
        for (int i = (backpointers.size() - 1); i >= 0; i--) {
            bestPath.addFirst(state);
            Map<String, String> backpointer = backpointers.get(i);
            state = backpointer.get(state);
        }
        return bestPath;
    }
}