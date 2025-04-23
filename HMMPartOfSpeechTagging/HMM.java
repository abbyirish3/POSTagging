import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.*;

/**
 * HMM model to tag parts of speech in a sentence
 *
 * @author Abigail Irish
 */
public class HMM {
    private Map<String, Map<String, Double>> transitionProbs;
    private Map<String, Map<String, Double>> observationProbs;
    private List<String[]> sentences;
    private List<String[]> tags;

    // constructor creates empty maps--call train to add to them
    public HMM() {
        transitionProbs = new HashMap<>();
        observationProbs = new HashMap<>();
    }

    // overloaded constructor creates maps with training files provided
    public HMM(String sentenceFile, String tagFile) throws IOException {
        transitionProbs = new HashMap<>();
        observationProbs = new HashMap<>();

        this.sentences = loadTrainingSentenceFile(sentenceFile);
        this.tags = loadTrainingTagFile(tagFile);
        train(sentences, tags);
    }

    /**
     * getters to return transition probabilities and observation probabilities
     *
     * @return Map of transitionProbs and observationProbs
     */
    public Map<String, Map<String, Double>> getTransitionProbs() {
        return transitionProbs;
    }

    public Map<String, Map<String, Double>> getObservationProbs() {
        return observationProbs;
    }

    /**
     * methods to load in the test sentences files and test tags files
     * @param fileName
     * @return List of sentences or tags
     * @throws IOException
     */
    public List<String[]> loadTrainingSentenceFile(String fileName) throws IOException {
        List<String[]> sentences = new ArrayList<>();
        try (
                BufferedReader input = new BufferedReader(new FileReader(fileName))) {
            String line;
            while ((line = input.readLine()) != null) {
                sentences.add(line.toLowerCase().split("\\s+"));
            }
            input.close();
        }
        return sentences;
    }

    public List<String[]> loadTrainingTagFile(String fileName) throws IOException {
        List<String[]> tags = new ArrayList<>();
        try (
                BufferedReader input = new BufferedReader(new FileReader(fileName))) {
            String line;
            while ((line = input.readLine()) != null) {
                tags.add(line.split("\\s+"));
            }
            input.close();
        }
        return tags;
    }

    /**
     * Trains the HMM on a list of sentences and their corresponding tag sequences.
     * Each sentence and tag sequence are represented as arrays of tokens.
     *
     * @param sentences    List of token arrays (each array is a sentence)
     * @param tagSequences List of tag arrays (each array is the corresponding tags)
     */
    public void train(List<String[]> sentences, List<String[]> tagSequences) {
        String startState = "#";
        Map<String, Map<String, Integer>> transitionCounts = new HashMap<>();
        Map<String, Map<String, Integer>> observationCounts = new HashMap<>();

        for (int i = 0; i < sentences.size(); i++) {
            String[] tokens = sentences.get(i);
            String[] tags = tagSequences.get(i);
            if (tokens.length != tags.length) {
                System.err.println("Sentence and tag sequence lengths do not match for sentence " + i);
                continue;
            }

            addCount(transitionCounts, startState, tags[0]);
            addCount(observationCounts, tags[0], tokens[0]);

            for (int j = 1; j < tokens.length; j++) {
                addCount(transitionCounts, tags[j - 1], tags[j]);
                addCount(observationCounts, tags[j], tokens[j]);
            }
        }
        // Convert counts to log probabilities.
        transitionProbs = convertCountsToLogProbs(transitionCounts);
        observationProbs = convertCountsToLogProbs(observationCounts);
    }

    /**
     * Helper method to increment a count in a nested map.
     * For example, addCount(transitionCounts, "NP", "V") increments the count of transitioning from NP to V.
     */
    private void addCount(Map<String, Map<String, Integer>> counts, String key, String subKey) {
        counts.putIfAbsent(key, new HashMap<>());
        Map<String, Integer> innerMap = counts.get(key);
        innerMap.put(subKey, innerMap.getOrDefault(subKey, 0) + 1);
    }

    /**
     * Converts a nested count map to a nested probability map (in log space).
     *
     * @param counts Nested map of counts.
     * @return Nested map of log probabilities.
     */
    private Map<String, Map<String, Double>> convertCountsToLogProbs(Map<String, Map<String, Integer>> counts) {
        Map<String, Map<String, Double>> logProbs = new HashMap<>();

        for (String key : counts.keySet()) {
            Map<String, Integer> innerCounts = counts.get(key);
            int total = 0;
            for (int count : innerCounts.values()) {
                total += count;
            }
            Map<String, Double> innerLogProbs = new HashMap<>();
            for (String subKey : innerCounts.keySet()) {
                double probability = (double) innerCounts.get(subKey) / total;
                innerLogProbs.put(subKey, Math.log(probability));
            }
            logProbs.put(key, innerLogProbs);
        }
        return logProbs;
    }

    /**
     * toString method to inspect the model's probabilities
     * @return
     */
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("Transition Probabilities:\n");
        for (String state : transitionProbs.keySet()) {
            sb.append(state).append(" -> ").append(transitionProbs.get(state)).append("\n");
        }
        sb.append("Observation Probabilities:\n");
        for (String tag : observationProbs.keySet()) {
            sb.append(tag).append(" -> ").append(observationProbs.get(tag)).append("\n");
        }
        return sb.toString();
    }

    /**
     * method to allow users to type a sentence as input
     * prints out the predicted tags for the input
     */
    public void consoleTest() {
        Scanner scanner = new Scanner(System.in);
        System.out.println("Please type your input sentence and press enter. \n" +
                "Type 'stop' when you want to finish the console test.");

        while (true) {
            String line = scanner.nextLine().trim(); // Read user input

            if (line.equalsIgnoreCase("stop")) { // If the user types 'stop', stop the loop
                System.out.println("Stopping console test...");
                scanner.close();
                break;
            }
            if (line.isEmpty()) { // If the user presses Enter without typing anything, stop
                break;
            }
            String[] sentence = line.toLowerCase().split("\\s+");

            Viterbi viterbi = new Viterbi(); // Assuming Viterbi has this constructor
            List<String> predictedTags = viterbi.decode(sentence, this); // Decode each sentence
            System.out.println("Predicted Tags for Sentence: " + String.join(" ", sentence));
            System.out.println("Predicted Tags: " + predictedTags);
        }
    }

    public void testFileAccuracy(HMM hmm) {
        int correct = 0, total = 0;
        Viterbi viterbi = new Viterbi();

        for (int i = 0; i < hmm.sentences.size(); i++) {
            String[] words = hmm.sentences.get(i);
            String[] trueTags = hmm.tags.get(i);
            List<String> predictedTags = viterbi.decode(words, hmm);

            for (int j = 0; j < words.length; j++) {
                if (predictedTags.get(j).equals(trueTags[j])) {
                    correct++;
                }
                total++;
            }
        }
        System.out.println("Number of correct tags: " + correct + "\n");
        System.out.println("Number of incorrect tags: " + (total - correct) + "\n");
        System.out.println("Accuracy: " + (100.0 * correct / total) + "%");
    }


    public static void main(String[] args) {
        // HARDCODED TEST CASE
        // Step 1: Manually Define Training Data (Words and Tags)
        List<String[]> hardCodedSentences = new ArrayList<>();
        List<String[]> hardCodedTags = new ArrayList<>();

        // Example training data: sentences and corresponding POS tags
        hardCodedSentences.add(new String[]{"the", "dog", "runs"});
        hardCodedTags.add(new String[]{"DET", "NOUN", "VERB"});

        hardCodedSentences.add(new String[]{"a", "cat", "sleeps"});
        hardCodedTags.add(new String[]{"DET", "NOUN", "VERB"});

        hardCodedSentences.add(new String[]{"the", "man", "eats"});
        hardCodedTags.add(new String[]{"DET", "NOUN", "VERB"});

        // Step 2: Train HMM directly with the given data
        HMM hmmHC = new HMM();
        hmmHC.train(hardCodedSentences, hardCodedTags);

        // Step 3: Define a test sentence (to be tagged)
        String testSentence = "a dog eats";
        String[] words = testSentence.toLowerCase().split("\\s+");

        // Step 4: Run the Viterbi algorithm
        Viterbi viterbi = new Viterbi();
        List<String> predictedTags = viterbi.decode(words, hmmHC); // Uses the manually trained model
        System.out.println("Sentence: " + testSentence);
        System.out.println("Predicted Tags: " + predictedTags);


        // FILE TEST CASES
        try {
            HMM hmm = new HMM("simple-test-sentences.txt", "simple-test-tags.txt");
            System.out.println("testing file: simple-test-sentences.txt");
            hmm.testFileAccuracy(hmm);

            HMM hmm2 = new HMM("simple-train-sentences.txt", "simple-train-tags.txt");
            System.out.println("testing file: simple-train-sentences.txt");
            hmm2.testFileAccuracy(hmm2);

            HMM hmm3 = new HMM("brown-test-sentences.txt", "brown-test-tags.txt");
            System.out.println("testing file: brown-test-sentences.txt");
            hmm3.testFileAccuracy(hmm3);

            HMM hmm4 = new HMM("brown-train-sentences.txt", "brown-train-tags.txt");
            System.out.println("testing file: brown-train-sentences.txt");
            hmm4.testFileAccuracy(hmm4);


            // CONSOLE TEST CASE
            System.out.println("Starting the console test...");
            hmm.consoleTest();  //let the user type in sentences to test
        }
        catch (IOException e) {
            System.out.println(e);
        }
    }
}