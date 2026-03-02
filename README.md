# Iris Classifier

A from-scratch classification pipeline built on the Fisher Iris dataset. The goal was not to achieve state-of-the-art results on a toy dataset -- it was to build a clean, modular ML pipeline that follows production patterns from day one.

---

## Architecture

```
load_data() --> split_data() --> train_model() --> evaluate_model() --> save_model()
                                                                            |
                                                                       [disk: .pkl]
                                                                            |
                                                         load_model() --> predict()
```

Each function is a self-contained primitive. No global state. Any step can be swapped without touching the rest.

---

## Results

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Setosa | 1.00 | 1.00 | 1.00 | 10 |
| Versicolor | 0.90 | 0.90 | 0.90 | 10 |
| Virginica | 0.90 | 0.90 | 0.90 | 10 |
| **Overall Accuracy** | | | **0.933** | **30** |

93.3% accuracy on a stratified 80/20 split with a fixed seed (`random_state=42`). Reproducible on any machine.

---

## Design Decisions

**Random Forest over alternatives.** 100 trees, `max_depth=5`. The depth cap is deliberate -- 150 samples across 3 classes does not justify a deep tree. Overfitting is the real risk here, not underfitting.

**Stratified split.** With only 50 samples per class, a naive random split can easily skew the test set. `stratify=y` guarantees balanced class representation in both train and test.

**Feature importance from the ensemble:**

| Feature | Importance |
|---------|-----------|
| Petal width | 0.438 |
| Petal length | 0.432 |
| Sepal length | 0.116 |
| Sepal width | 0.014 |

Petal measurements carry ~87% of the signal. Sepal width is nearly irrelevant. This aligns with the known biology -- petal morphology is the primary differentiator between Iris species.

**Pickle serialization for persistence.** The trained model is written to disk as a `.pkl` file (153 KB). This separates training from inference -- you train once, then load and predict without retraining.

---

## Project Structure

```
iris_classifier/
    iris_classifier.ipynb   # Full pipeline: load, split, train, evaluate, save, predict
    models/
        iris_model.pkl      # Trained RandomForest (serialized)
    data/                   # Reserved for external datasets
    README.md
```

---

## Usage

**Train and evaluate:**

```python
X, y = load_data()
X_train, X_test, y_train, y_test = split_data(X, y)
model = train_model(X_train, y_train)
accuracy = evaluate_model(model, X_test, y_test)
save_model(model)
```

**Load and predict:**

```python
model = load_model()
sample = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns=X.columns)
prediction = predict(model, sample)
# Output: Setosa (class 0), probability 1.00
```

---

## Dependencies

- Python 3.x
- scikit-learn
- pandas
- numpy

---

## What This Demonstrates

- Modular pipeline design with isolated primitives
- Proper train/test methodology (stratified split, fixed seed, reproducibility)
- Model persistence and separation of training from inference
- Feature importance analysis to understand what the model actually learned
- Deliberate hyperparameter choices with reasoning, not defaults

---

*Part of a series building ML models from the ground up. Each project adds complexity while keeping the engineering fundamentals tight.*
