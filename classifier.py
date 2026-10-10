import math
from collections import defaultdict


class NaiveBayesClassifier:
    def __init__(self):
        self.log_class_priors = {}
        self.word_counts = defaultdict(lambda: defaultdict(int))
        self.vocab = set()
        self.class_word_totals = defaultdict(int)

    def train(self, data):
        """
        Trains the Naive Bayes classifier.
        data: A list of tuples, e.g.,
        [("This movie is great", "positive"), ...]
        """
        class_counts = defaultdict(int)
        total_docs = len(data)

        # Count frequencies
        for text, label in data:
            class_counts[label] += 1
            words = text.lower().split()

            for word in words:
                self.word_counts[label][word] += 1
                self.class_word_totals[label] += 1
                self.vocab.add(word)

        # Calculate log priors
        for label, count in class_counts.items():
            self.log_class_priors[label] = math.log(
                count / total_docs
            )

    def predict(self, text):
        """
        Predicts the class for a given text.
        Returns the class with the highest probability.
        """
        words = text.lower().split()
        best_label = None
        max_log_prob = float("-inf")

        vocab_size = len(self.vocab)

        for label in self.log_class_priors:
            # Start with log prior
            log_prob = self.log_class_priors[label]

            for word in words:
                # Add Laplace (add-1) smoothing
                count = self.word_counts[label].get(
                    word,
                    0
                )

                # P(word|class) =
                # (count + 1) /
                # (total_words_in_class + vocab_size)
                word_prob = (
                    (count + 1)
                    / (
                        self.class_word_totals[label]
                        + vocab_size
                    )
                )

                log_prob += math.log(word_prob)

            if log_prob > max_log_prob:
                max_log_prob = log_prob
                best_label = label

        return best_label


if __name__ == "__main__":
    # A small synthetic dataset of movie reviews
    training_data = [
        (
            "I loved this movie it was fantastic",
            "positive"
        ),
        (
            "Great acting and beautiful cinematography",
            "positive"
        ),
        (
            "What a wonderful experience highly recommend",
            "positive"
        ),
        (
            "The plot was amazing and thrilling",
            "positive"
        ),
        (
            "I hated the movie it was terrible",
            "negative"
        ),
        (
            "Awful acting and a boring plot",
            "negative"
        ),
        (
            "What a waste of time",
            "negative"
        ),
        (
            "I would not recommend this movie to anyone",
            "negative"
        )
    ]

    print(
        "Initializing and training "
        "Naive Bayes Classifier from scratch..."
    )

    nb = NaiveBayesClassifier()
    nb.train(training_data)

    print(
        f"Training complete. "
        f"Vocabulary size: {len(nb.vocab)}"
    )

    print("\n--- Interactive Sentiment Classifier ---")
    print(
        "Enter a movie review and the program "
        "will predict POSITIVE or NEGATIVE."
    )
    print("Type 'quit' to exit.")

    while True:
        review = input("\nEnter a movie review: ").strip()

        if review.lower() == "quit":
            print("Goodbye!")
            break

        if not review:
            print("Please enter a review.")
            continue

        prediction = nb.predict(review)

        print(
            f"Predicted Sentiment: "
            f"{prediction.upper()}"
        )
