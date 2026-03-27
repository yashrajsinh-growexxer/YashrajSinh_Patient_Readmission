# Decision log

This file documents three key decision points from your pipeline.
**Each entry is mandatory. Vague or generic answers will be penalised.**
The question to answer is not "what does this technique do" – it is "why did YOU make THIS choice given YOUR data."

---

## Decision 1: Data cleaning strategy
*Complete after Phase 1 (by approximately 1:30)*

**What I did:**
Replaced `age` values of 999 with NaN, converted `blood_pressure_systolic` values < 50 from kPa to mmHg, and used median imputation for missing `age` and `glucose_level_mgdl` values evaluated precisely from the training split. Importantly, I added a binary `glucose_missing` indicator feature before imputing missing glucose levels.

**Why I did it (not why the technique works – why THIS choice given what you observed in the data):**
Because missing values in clinical data (like glucose tests) are rarely random—missing a test often indicates the patient's condition was not deemed severe enough to warrant it, meaning the *absence* of the test carries predictive clinical signal. 999 was an obvious sentinel value that distorted the age distribution. I constrained the median calculation strictly to the training set to prevent data leakage into the validation/test splits.

**What I considered and rejected:**
I considered dropping rows with missing glucose levels or using mean imputation. I rejected dropping rows because dropping ~20% of a small dataset (3800 rows) causes unacceptable information loss. I rejected mean imputation because clinical metrics are often heavily skewed rightwards by critical patients, making the median a more robust descriptor of a "typical" patient.

**What would happen if I was wrong here:**
If my BP unit assumption was wrong, I would artificially inflate the BP of patients genuinely suffering from severe hypotension, potentially masking a critical indicator for readmission. If the missingness of glucose was purely random (MCAR), the `glucose_missing` column would just be noise.

---

## Decision 2: Model architecture and handling class imbalance
*Complete after Phase 2 (by approximately 3:00)*

**What I did:**
I built a 4-layer fully-connected neural network (128->64->32->16) with BatchNorm and high Dropout (0.3/0.2) in PyTorch. For the severe 10:1 class imbalance, I applied a three-pronged strategy: 1) SMOTE to oversample the minority class in the training set to a 0.5 ratio, 2) Class-weighted Binary Cross-Entropy (BCE) loss during training, and 3) Decision threshold tuning post-training.

**Why I did it (not why the technique works – why THIS choice given what you observed in the data):**
The dataset size is extremely small for deep learning (3800 rows) but highly imbalanced (9% readmission). Deep networks naturally overfit this size, making heavy regularisation (Dropout + BatchNorm) mandatory to force the network to learn generalized patterns rather than memorizing the few minority instances. A single imbalance mitigation technique proved insufficient, so padding the feature space with SMOTE while strictly penalizing minority misclassifications with weighted BCE forced the network to pay attention to the positive class.

**What I considered and rejected:**
I considered Random Undersampling of the majority class. I rejected this because downsampling the already tiny 3800-row dataset would leave the network with barely 600 total observations to train on, causing severe underfitting and discarding valuable variance in the majority class.

**What would happen if I was wrong here:**
If my Dropout rates were too aggressive, the model would underfit and fail to learn complex interactions between clinical features. If SMOTE generated unrealistic synthetic patient profiles, it would heavily warp the decision boundary, hurting generalisation to the un-SMOTEd test set.

---

## Decision 3: Evaluation metric and threshold selection
*Complete after Phase 3 (by approximately 4:00)*

**What I did:**
I discarded standard Accuracy during evaluation and strictly monitored AUC-ROC and AUC-PR to gauge model generalisation. Post-training, rather than using the default 0.5 probability threshold, I scanned thresholds from 0.05 to 0.90 to find the exact cut-off that maximized the F1-score on the validation set independently.

**Why I did it (not why the technique works – why THIS choice given what you observed in the data):**
In a dataset with a 91% majority class, a completely useless "dummy" model that predicts "Not Readmitted" for every single patient achieves 91% accuracy automatically. Therefore, accuracy is deeply misleading. F1-score balances precision and recall, allowing me to find a practical operational point. The output probabilities of imbalanced learning are generally squashed, so the default 0.5 threshold is structurally incorrect; tuning it directly against F1-score ensures optimal class separability.

**What I considered and rejected:**
I considered optimizing solely for Recall (Sensitivity). While clinically it is vital not to miss readmissions (false negatives), optimizing purely for Recall collapses Precision. This results in overwhelming false positive alerts, leading to "alert fatigue" where hospital staff begin ignoring the model's predictions entirely. F1 prevents this collapse.

**What would happen if I was wrong here:**
If the validation set was not perfectly representative of the test data, the threshold highly optimized for the validation F1-score would be unstable and cause performance degradation when encountering the real test distribution.
