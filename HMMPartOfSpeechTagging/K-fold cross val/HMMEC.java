import java.io.*;
import java.util.*;

/**
 * HMM code to split dataset into partitions
 * and implement cross-validation
 *
 * @author Abigail Irish
 */
public class HMMEC {
    private Map<String, Map<String, Double>> transitionProbs;
    private Map<String, Map<String, Double>> observationProbs;
    private List<String[]> sentences;
    private List<String[]> tags;

    // constructor creates empty maps--call train to add to them
    public HMMEC() {
        transitionProbs = new HashMap<>();
        observationProbs = new HashMap<>();
    }

    // overloaded constructor creates maps with training files provided
    public HMMEC(String sentenceFile, String tagFile) throws IOException {
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
    private List<String[]> loadTrainingSentenceFile(String fileName) throws IOException {
        List<String[]> sentences = new ArrayList<>();
        try (BufferedReader input = new BufferedReader(new FileReader(fileName))) {
            String line;
            while ((line = input.readLine()) != null) {
                sentences.add(line.toLowerCase().split("\\s+"));
            }
        }
        return sentences;
    }

    private List<String[]> loadTrainingTagFile(String fileName) throws IOException {
        List<String[]> tags = new ArrayList<>();
        try (BufferedReader input = new BufferedReader(new FileReader(fileName))) {
            String line;
            while ((line = input.readLine()) != null) {
                tags.add(line.split("\\s+"));
            }
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
     * method to implement cross-validation: each part is used as a test set,
     * with the other parts used in training, to construct the model.
     *
     * @param k number of folds
     */
    public void crossValidate(int k) {
        if (sentences.size() != tags.size()) {
            System.err.println("Error: Sentence and tag files do not match in length.");
            return;
        }

        // Shuffle the data to avoid genre bias
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < sentences.size(); i++) {
            indices.add(i);
        }
        Collections.shuffle(indices, new Random(33));

        // Partition data into k folds
        List<List<String[]>> sentenceFolds = new ArrayList<>();
        List<List<String[]>> tagFolds = new ArrayList<>();
        for (int i = 0; i < k; i++) {
            sentenceFolds.add(new ArrayList<>());
            tagFolds.add(new ArrayList<>());
        }

        for (int i = 0; i < indices.size(); i++) {
            int fold = i % k;
            sentenceFolds.get(fold).add(sentences.get(indices.get(i)));
            tagFolds.get(fold).add(tags.get(indices.get(i)));
        }

        // Perform k-fold cross-validation
        double totalAccuracy = 0.0;
        for (int fold = 0; fold < k; fold++) {
            System.out.println("\nRunning Fold " + (fold + 1) + "/" + k);

            List<String[]> trainSentences = new ArrayList<>();
            List<String[]> trainTags = new ArrayList<>();
            List<String[]> testSentences = sentenceFolds.get(fold);
            List<String[]> testTags = tagFolds.get(fold);

            for (int i = 0; i < k; i++) {
                if (i == fold) continue; // Skip test set
                trainSentences.addAll(sentenceFolds.get(i));
                trainTags.addAll(tagFolds.get(i));
            }

            // Train and test
            train(trainSentences, trainTags);
            double accuracy = testAccuracy(testSentences, testTags);
            totalAccuracy += accuracy;
            System.out.println("Fold " + (fold + 1) + " Accuracy: " + accuracy + "%");
        }

        System.out.println("\nAverage Cross-Validation Accuracy: " + (totalAccuracy / k) + "%");
    }

    // Test accuracy function
    private double testAccuracy(List<String[]> testSentences, List<String[]> testTags) {
        int correct = 0, total = 0;
        ViterbiEC viterbiEC = new ViterbiEC();

        for (int i = 0; i < testSentences.size(); i++) {
            String[] words = testSentences.get(i);
            String[] trueTags = testTags.get(i);
            List<String> predictedTags = viterbiEC.decodeEC(words, this);

            for (int j = 0; j < words.length; j++) {
                if (j < predictedTags.size() && predictedTags.get(j).equals(trueTags[j])) {
                    correct++;
                }
                total++;
            }
        }

        return (100.0 * correct / total);
    }

    public static void main(String[] args) {
        try {
            HMMEC hmmEC = new HMMEC("brown-test-sentences.txt", "brown-test-tags.txt");
            hmmEC.crossValidate(5);
        } catch (IOException e) {
            System.out.println("Error loading data: " + e.getMessage());
        }
    }
}
