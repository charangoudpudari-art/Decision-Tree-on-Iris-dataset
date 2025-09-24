PUDARI SREE RAM CHARAN TEJA 700762701

# Decision Tree Classifier on Iris Dataset

## Project Overview
This project demonstrates how to apply a **Decision Tree Classifier** from `scikit-learn` on the well-known **Iris dataset**.  
The main goal is to show how changing the **depth of the decision tree** affects the model’s performance, and to illustrate concepts of **underfitting vs. overfitting**.  

---

## What is a Decision Tree?
A **Decision Tree** is a supervised machine learning model that splits data into branches to make predictions.  
- It selects features and thresholds to best separate classes.  
- Each split tries to reduce impurity (using Gini index or entropy).  
- Trees can become very deep and perfectly memorize training data → **overfitting**.  
- Trees that are too shallow may miss important patterns → **underfitting**.  

---

## Steps Performed in Code
1. **Load Dataset**:  
   - The Iris dataset has 150 samples of flowers, each with 4 features:  
     *Sepal length, Sepal width, Petal length, Petal width*.  
   - Target = flower species (`setosa`, `versicolor`, `virginica`).  

2. **Split Data**:  
   - Training set = 70%  
   - Test set = 30%  

3. **Train Classifier**:  
   - Train 3 decision trees with different depths (`max_depth = 1, 2, 3`).  

4. **Evaluate**:  
   - Report **training accuracy** (how well the model fits known data).  
   - Report **test accuracy** (how well the model generalizes to unseen data).  

5. **Visualize**:  
   - Use `plot_tree` from sklearn to show how the model splits features at each depth.  

---

## 📊 Results (Expected Output)

| Max Depth | Training Accuracy | Test Accuracy | Notes |
|-----------|-------------------|---------------|-------|
| 1         | ~0.73             | ~0.71         | **Underfitting** – model too simple |
| 2         | ~0.95             | ~0.93         | **Good Fit** – best generalization |
| 3         | ~0.97             | ~0.93         | **Overfitting begins** – no gain in test accuracy
## Key Concepts
- **Underfitting** (Depth = 1):  
  The tree is too shallow, misses patterns, poor performance on both train & test.  

- **Overfitting** (Depth too large):  
  The tree memorizes training data (very high training accuracy) but does not generalize well.  

- **Good Fit** (Depth = 2–3 here):  
  Balanced model that performs well on both train and test sets.  

---

## Code Explanation
- `DecisionTreeClassifier(max_depth=d)` → Build tree limited to depth `d`.  
- `fit(X_train, y_train)` → Train on training data.  
- `predict(X)` → Predict labels for inputs.  
- `accuracy_score(y_true, y_pred)` → Measure accuracy.  
- `plot_tree(clf)` → Draw the trained tree structure.  

---

## ▶️ How to Run

### 1. Install dependencies
```bash
pip install scikit-learn matplotlib
