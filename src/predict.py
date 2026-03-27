import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Hardcode the model architecture to match what was trained in the notebook
class ReadmissionNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).squeeze()

def load_and_preprocess(test_path, train_path='data/train.csv'):
    # Load data
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Store patient IDs
    patient_ids = test_df['patient_id'].copy() if 'patient_id' in test_df.columns else None
    
    # Drop irrelevant
    train_df.drop(columns=['patient_id'], inplace=True, errors='ignore')
    test_df.drop(columns=['patient_id'], inplace=True, errors='ignore')

    # Outliers
    train_df['age'] = train_df['age'].replace(999, np.nan)
    test_df['age'] = test_df['age'].replace(999, np.nan)
    
    mask_train = train_df['blood_pressure_systolic'] < 50
    train_df.loc[mask_train, 'blood_pressure_systolic'] *= 7.50062
    mask_test = test_df['blood_pressure_systolic'] < 50
    test_df.loc[mask_test, 'blood_pressure_systolic'] *= 7.50062

    # Dates
    for df in [train_df, test_df]:
        if 'admission_date' in df.columns:
            df['admission_date'] = pd.to_datetime(df['admission_date'], format='mixed', dayfirst=True, errors='coerce')
            df['admission_month'] = df['admission_date'].dt.month
            df['admission_day'] = df['admission_date'].dt.day
            df['admission_weekday'] = df['admission_date'].dt.weekday
            df.drop(columns=['admission_date'], inplace=True)

    # Missing indicator & Median Imputation (fit on train)
    train_df['glucose_missing'] = train_df['glucose_level_mgdl'].isnull().astype(int)
    test_df['glucose_missing'] = test_df['glucose_level_mgdl'].isnull().astype(int)
    
    age_median = train_df['age'].median()
    glucose_median = train_df['glucose_level_mgdl'].median()
    
    train_df['age'].fillna(age_median, inplace=True)
    test_df['age'].fillna(age_median, inplace=True)
    train_df['glucose_level_mgdl'].fillna(glucose_median, inplace=True)
    test_df['glucose_level_mgdl'].fillna(glucose_median, inplace=True)

    # Encode Gender
    train_df['gender'] = (train_df['gender'] == 'M').astype(int)
    test_df['gender'] = (test_df['gender'] == 'M').astype(int)

    # One-Hot Encoding aligned to Train
    train_ins = pd.get_dummies(train_df['insurance_type'], prefix='ins')
    test_ins = pd.get_dummies(test_df['insurance_type'], prefix='ins')
    for col in train_ins.columns:
        if col not in test_ins.columns: test_ins[col] = 0
    test_ins = test_ins[train_ins.columns]
    test_df = pd.concat([test_df.drop(columns=['insurance_type']), test_ins], axis=1)
    train_df = pd.concat([train_df.drop(columns=['insurance_type']), train_ins], axis=1)

    train_day = pd.get_dummies(train_df['discharge_day_of_week'], prefix='day')
    test_day = pd.get_dummies(test_df['discharge_day_of_week'], prefix='day')
    for col in train_day.columns:
        if col not in test_day.columns: test_day[col] = 0
    test_day = test_day[train_day.columns]
    test_df = pd.concat([test_df.drop(columns=['discharge_day_of_week']), test_day], axis=1)
    train_df = pd.concat([train_df.drop(columns=['discharge_day_of_week']), train_day], axis=1)

    TARGET = 'readmitted_30d'
    train_features = train_df.drop(columns=[TARGET]) if TARGET in train_df.columns else train_df
    
    for col in train_features.columns:
        if col not in test_df.columns:
            test_df[col] = 0
    test_df = test_df[train_features.columns]

    # Scaling
    numeric_cols = [
        "age", "admission_type", "discharge_destination",
        "length_of_stay_days", "charlson_comorbidity_index",
        "prior_admissions_1yr", "n_medications_discharge",
        "glucose_level_mgdl", "blood_pressure_systolic",
        "sodium_meql", "creatinine_mgdl", "haemoglobin_gdl",
        "admission_month", "admission_day", "admission_weekday"
    ]
    numeric_cols = [c for c in numeric_cols if c in train_features.columns]

    scaler = StandardScaler()
    scaler.fit(train_features[numeric_cols])
    test_df[numeric_cols] = scaler.transform(test_df[numeric_cols])

    return test_df.values.astype(np.float32), patient_ids

def main():
    parser = argparse.ArgumentParser(description='Inference Script for 30-Day Readmission')
    parser.add_argument('--input', type=str, required=True, help='Path to test.csv')
    parser.add_argument('--output', type=str, default='predictions.csv', help='Path to output csv')
    args = parser.parse_args()

    print(f"Loading data from {args.input}...")
    X_test, patient_ids = load_and_preprocess(args.input)
    
    input_dim = X_test.shape[1]
    model = ReadmissionNet(input_dim)
    
    print("Loading model weights from best_model.pth...")
    try:
        model.load_state_dict(torch.load('best_model.pth', map_location=torch.device('cpu')))
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure 'best_model.pth' is in the root directory.")
        return
        
    model.eval()
    
    X_t = torch.FloatTensor(X_test)
    print("Running inference...")
    with torch.no_grad():
        probs = model(X_t).numpy()
    
    # 0.20 was typically an optimal decision threshold for F1 on this dataset
    best_thresh = 0.20
    preds = (probs >= best_thresh).astype(int)
    
    print(f"Completed! {preds.sum()} patients predicted to readmit.")
    
    if patient_ids is not None:
        out_df = pd.DataFrame({
            'patient_id': patient_ids,
            'readmission_probability': probs
        })
    else:
        out_df = pd.DataFrame({
            'readmission_probability': probs
        })
        
    out_df.to_csv(args.output, index=False)
    print(f"Results saved to {args.output}")

if __name__ == '__main__':
    main()
