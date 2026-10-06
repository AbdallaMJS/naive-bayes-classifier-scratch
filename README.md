# Naive Bayes Classifier from Scratch

A custom-built Multinomial Naive Bayes text classifier implemented entirely in Python without relying on heavy machine learning frameworks like `scikit-learn` or `TensorFlow`. This project demonstrates a deep understanding of natural language processing (NLP) basics, probability theory (Bayes' Theorem), and data processing.

## 🌟 Key Features

- **Built from the Ground Up:** Implements the core logic of Naive Bayes classification entirely using Python's standard libraries (`math`, `collections`).
- **Laplace (Add-1) Smoothing:** Prevents the "zero probability" problem encountered when the model evaluates unseen words during the prediction phase, ensuring robust and mathematically sound outputs.
- **Log Probabilities:** Uses logarithmic probabilities to prevent arithmetic underflow, a common issue in machine learning when multiplying many small probability fractions.
- **Dynamic Vocabulary Construction:** Automatically builds its vocabulary and class priors based on the provided training data, making it adaptable to various text classification tasks beyond just sentiment analysis.

## 🛠️ Technologies Used

- **Language:** Python
- **Core Libraries:** `math` (for logarithms), `collections` (for dictionaries and counting)

## 💡 How It Works (The Math)

The classifier operates on **Bayes' Theorem**:
`P(A|B) = [P(B|A) * P(A)] / P(B)`

For text classification, this translates to finding the probability of a *Class* (positive/negative) given a *Document* (a set of words).
We calculate:
`P(Class | Document) ∝ P(Class) * Π P(Word_i | Class)`

1. **Training Phase:** Calculates the Prior Probability `P(Class)` for each label, and the Likelihood `P(Word | Class)` for every word in the vocabulary, factoring in Laplace smoothing.
2. **Prediction Phase:** Converts the probabilities to logarithms to add them together (instead of multiplying) to avoid underflow. It returns the class that yields the highest log-probability for the given text.

## 🚀 Setup and Execution

This project is fully self-contained and requires no external installations.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AbdallaMJS/naive-bayes-classifier-scratch.git
   cd naive-bayes-classifier-scratch
   ```

2. **Run the classifier:**
   ```bash
   python classifier.py
   ```
   The script includes a small synthetic training dataset and will output predictions for several test strings.

## 🧠 Educational Value & MBZUAI Relevance

This project is tailored to demonstrate early competency for advanced AI programs. While utilizing libraries like `scikit-learn` is standard in the industry, building fundamental algorithms from scratch proves a candidate understands the underlying mathematics and statistics of Machine Learning, not just the API calls. Understanding Naive Bayes is a crucial stepping stone before moving into complex Deep Learning models for NLP.

---
*Developed by Abdalla M.J.S. Alblooshi to showcase foundational knowledge in probability, machine learning mathematics, and Python software engineering.*
