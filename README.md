# Readmission-DL — City General Hospital 30-day Readmission Prediction

**name:** Yashraj Sinh



---

## Problem

Predict whether a patient will be readmitted within 30 days of discharge using structured clinical data from City General Hospital (3,800 training records, 950 test records).

---

## My model

**Architecture:**
<!-- Describe your network: layer sizes, activations, regularisation -->
The model is a 4-layer Fully Connected Neural Network built in PyTorch (128 -> 64 -> 32 -> 16 -> 1). Between each hidden layer, I utilized ReLU activations. Due to the small size of the dataset (3,800 records), heavy regularization was essential to prevent rapid overfitting; therefore, I inserted `BatchNorm1d` and `Dropout` (0.3 and 0.2 rates) after the first three layers. The output is a single neuron with a Sigmoid activation.

**Key preprocessing decisions:**
<!-- Summarise the most important choices – 2-3 sentences -->
Firstly, I extracted standard date elements (month/day/weekday) from the admission dates and replaced known outlier ages (`999`) with NaNs. Missing tests like `glucose_level_mgdl` inherently possess non-random informational value, so I explicitly added a binary `glucose_missing` indicator before applying train-set targeted median imputation. Finally, blood pressure values < 50 were scaled back up (kPa -> mmHg conversion error), and standard `LabelEncoder` / One-Hot encoding was aligned to prevent data leakage.

**How I handled class imbalance:**
<!-- What technique and why -->
The dataset features a severe 91% vs 9% class imbalance, making a standard binary cross-entropy (BCE) model artificially biased heavily towards the majority class. I countered this with three techniques: 
1. **SMOTE** oversampling on the training split to synthesize minority samples to 50% of the majority.
2. Applying **Class-Weighted BCE Loss** dynamically scaled during training.
3. Completely ignoring Accuracy as a useless metric, instead dynamically tuning the model decision threshold directly against Validation F1-scores.

---



## How to run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model (optional – pretrained weights included)

```bash
# Assuming you have jupyter installed
python -m jupyter nbconvert --to notebook --execute notebooks/solution.ipynb

# Or interactively run the cells in `notebooks/solution.ipynb` in order.
```

### 3. Run inference on the test set

Using the self-contained PyTorch prediction script which seamlessly executes the same preprocessing pipeline dynamically:

```bash
python src/predict.py --input data/test.csv --output predictions.csv
```

The output CSV will contain two columns: `patient_id` and `readmission_probability`.
