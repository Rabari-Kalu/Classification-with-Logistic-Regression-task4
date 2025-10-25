# Classification-with-Logistic-Regression-task4
This project applies Logistic Regression to predict whether breast cancer is malignant (M) or benign (B) using the Breast Cancer Wisconsin Diagnostic Dataset. The model uses numeric cell nucleus features to make predictions and evaluates performance using key metrics such as precision, recall, ROC-AUC, and the confusion matrix.


from pathlib import Path

# Create README.md content again
readme_content = """# Logistic Regression - Binary Classification (Breast Cancer Detection)

##  Project Overview
This project implements a **binary classification model** using **Logistic Regression** on the **Breast Cancer Wisconsin (Diagnostic) dataset**.  
The goal is to predict whether a tumor is **Malignant (M)** or **Benign (B)** based on various features extracted from breast cell images.

---

##  Dataset Information
- **Dataset file:** `data.csv`
- **Source:** UCI Machine Learning Repository  
- **Rows:** 569  
- **Columns:** 33 (30 features + 1 target + 2 unnecessary columns)

###  Target Column:
- `diagnosis` →  
  - `M` = Malignant (1)  
  - `B` = Benign (0)

###  Dropped Columns:
- `id` – identifier  
- `Unnamed: 32` – empty column  

---

## ⚙️ Workflow

###  Import and Explore Data
- Loaded and inspected dataset using pandas  
- Checked for null values and removed unnecessary columns  

###  Data Preprocessing
- Encoded target variable: `M → 1`, `B → 0`  
- Standardized numerical features using `StandardScaler`

### Train-Test Split
- Split data into **80% train** and **20% test** using `train_test_split`

### Model Training
- Used **Logistic Regression** from `sklearn.linear_model`  
- Fit the model on scaled features  

### Evaluation Metrics
- **Confusion Matrix**
- **Precision**, **Recall**, **Accuracy**, **ROC-AUC**
- **Classification Report**
- **ROC Curve Visualization**
- **Threshold tuning** (tested with 0.6)

---

## Results

| Metric        | Typical Value |
|----------------|----------------|
| Accuracy       | ~0.96 |
| Precision      | ~0.97 |
| Recall         | ~0.95 |
| ROC-AUC        | ~0.99 |

### Key Plots:
- Confusion Matrix (heatmap)
- ROC-AUC Curve (True Positive Rate vs False Positive Rate)

---

##  Questions & Answers

** How does logistic regression differ from linear regression?**  
- Linear regression predicts **continuous values**, logistic regression predicts **probabilities (0–1)**.  
- Logistic regression uses the **sigmoid function** for classification.

**What is the sigmoid function?**  
It converts any real number into a probability between 0 and 1:  
\\[ \\sigma(x) = \\frac{1}{1 + e^{-x}} \\]  
Used to estimate the probability of belonging to a particular class.

** What is precision vs recall?**
- **Precision** → TP / (TP + FP): Correct positive predictions  
- **Recall** → TP / (TP + FN): Ability to find all positive cases  
- High precision reduces **false positives**, high recall reduces **false negatives**

** What is the ROC-AUC curve?**  
- ROC: Plot of **True Positive Rate vs False Positive Rate**  
- AUC: Area under ROC curve (measures separability)  
- Closer to 1 = better model performance

** What is the confusion matrix?**  
|                | Predicted Positive | Predicted Negative |
|----------------|--------------------|--------------------|
| **Actual Positive** | TP  | FN |
| **Actual Negative** | FP | TN |

** What happens if classes are imbalanced?**  
- Model can get biased towards majority class.  
- Solutions:  
  - Resampling (SMOTE, undersampling)  
  - Class weights (`class_weight='balanced'`)  
  - Use metrics like ROC-AUC, F1-score

** How do you choose the threshold?**  
- Default = 0.5  
- Adjust based on use case:  
  - Lower → higher recall  
  - Higher → higher precision  

** Can logistic regression be used for multi-class problems?**  
Yes, via:
- **One-vs-Rest (OvR)**
- **Multinomial (Softmax)**

---

## Tech Stack
- **Python**
- **Pandas**, **NumPy**
- **Scikit-learn**
- **Matplotlib**, **Seaborn**

---


