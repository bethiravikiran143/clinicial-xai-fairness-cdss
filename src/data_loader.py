"""
Clinical Dataset Loader and Preprocessor
Domain: Cardiology Clinical Decision Support System (CDSS)
Includes demographic sensitive attributes for fairness auditing: 'sex' and 'age_group'.
"""

import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DATA_FILE = os.path.join(DATA_DIR, "clinical_heart_data.csv")

def generate_clinical_cohort(n_samples=1200, random_state=42):
    """
    Generates a realistic clinical cohort reflecting cardiovascular hemodynamics
    and demographic distributions found in clinical literature (Cleveland/Framingham).
    Embeds realistic subtle diagnostic disparities across sex and age groups.
    """
    np.random.seed(random_state)

    # Demographic features
    # Sex: 1 = Male (60%), 0 = Female (40%)
    sex = np.random.choice([0, 1], size=n_samples, p=[0.42, 0.58])
    
    # Age: Normal distribution centered around 54 years
    age = np.clip(np.random.normal(loc=55, scale=10, size=n_samples), 29, 78).astype(int)
    
    # Chest Pain type: 0: Typical Angina, 1: Atypical Angina, 2: Non-anginal, 3: Asymptomatic
    # Note: Clinical literature shows females often report atypical or non-anginal symptoms
    cp_probs_male = [0.18, 0.22, 0.28, 0.32]
    cp_probs_female = [0.10, 0.35, 0.35, 0.20]
    cp = np.array([
        np.random.choice([0, 1, 2, 3], p=cp_probs_male if s == 1 else cp_probs_female)
        for s in sex
    ])

    # Resting Blood Pressure (trestbps) in mmHg: 94 - 200
    trestbps = np.clip(np.random.normal(loc=131 + 0.3 * (age - 50), scale=17, size=n_samples), 94, 200).astype(int)

    # Serum Cholesterol (chol) in mg/dl: 126 - 564
    chol = np.clip(np.random.normal(loc=246 + (15 if sex.mean() > 0.5 else 0), scale=51, size=n_samples), 126, 564).astype(int)

    # Fasting Blood Sugar > 120 mg/dl (fbs): 1 = True, 0 = False (~15% prevalence)
    fbs = np.random.choice([0, 1], size=n_samples, p=[0.85, 0.15])

    # Resting ECG (restecg): 0: Normal, 1: ST-T wave abnormality, 2: Left ventricular hypertrophy
    restecg = np.random.choice([0, 1, 2], size=n_samples, p=[0.50, 0.46, 0.04])

    # Maximum Heart Rate achieved (thalach): inversely correlated with age
    thalach = np.clip(np.random.normal(loc=205 - 0.9 * age, scale=22, size=n_samples), 71, 202).astype(int)

    # Exercise Induced Angina (exang): 1 = Yes, 0 = No
    exang_prob = np.clip(0.15 + (age - 40) * 0.006 + (cp == 3) * 0.2, 0.05, 0.85)
    exang = (np.random.rand(n_samples) < exang_prob).astype(int)

    # ST Depression induced by exercise (oldpeak): 0.0 to 6.2
    oldpeak = np.clip(np.random.exponential(scale=1.0, size=n_samples) * (1.0 + 0.4 * exang), 0.0, 6.2).round(1)

    # Slope of peak exercise ST segment: 0: Upsloping, 1: Flat, 2: Downsloping
    slope = np.random.choice([0, 1, 2], size=n_samples, p=[0.46, 0.46, 0.08])

    # Major vessels (ca) colored by fluoroscopy: 0 to 3
    ca_probs = [0.58, 0.22, 0.13, 0.07]
    ca = np.random.choice([0, 1, 2, 3], size=n_samples, p=ca_probs)

    # Thalassemia (thal): 1 = Normal, 2 = Fixed defect, 3 = Reversible defect
    thal = np.random.choice([1, 2, 3], size=n_samples, p=[0.55, 0.10, 0.35])

    # Clinical Risk Scoring / Ground Truth Target derivation
    # Log-odds of heart disease based on validated cardiology risk factors
    logits = (
        -4.2
        + 0.045 * (age - 50)
        + 0.55 * sex
        + 0.65 * (cp == 3)
        + 0.35 * (cp == 2)
        + 0.015 * (trestbps - 120)
        + 0.004 * (chol - 200)
        + 0.45 * fbs
        + 0.30 * (restecg > 0)
        - 0.025 * (thalach - 150)
        + 0.85 * exang
        + 0.60 * oldpeak
        + 0.40 * (slope == 1)
        + 0.70 * (ca > 0)
        + 0.80 * (thal == 3)
    )

    probs = 1.0 / (1.0 + np.exp(-logits))
    target = (np.random.rand(n_samples) < probs).astype(int)

    df = pd.DataFrame({
        "age": age,
        "sex": sex,                       # 1 = Male, 0 = Female
        "chest_pain_type": cp,            # 0: Typical, 1: Atypical, 2: Non-anginal, 3: Asymptomatic
        "resting_bp": trestbps,           # mm Hg
        "cholesterol": chol,              # mg/dl
        "fasting_bs": fbs,                # 1 if > 120 mg/dl else 0
        "resting_ecg": restecg,           # 0: Normal, 1: ST-T wave, 2: LVH
        "max_heart_rate": thalach,        # bpm
        "exercise_angina": exang,         # 1 = Yes, 0 = No
        "st_depression": oldpeak,         # mm
        "st_slope": slope,                # 0: Upsloping, 1: Flat, 2: Downsloping
        "num_major_vessels": ca,          # 0-3
        "thalassemia": thal,              # 1: Normal, 2: Fixed defect, 3: Reversible defect
        "heart_disease": target           # 0 = No, 1 = Yes
    })

    # Add human-interpretable sensitive cohort columns for auditing
    df["gender_label"] = df["sex"].map({1: "Male", 0: "Female"})
    df["age_group"] = np.where(df["age"] >= 55, "Senior (>=55)", "Adult (<55)")

    return df

def get_clinical_data(force_regenerate=False):
    """Loads clinical dataset from CSV or generates and saves if missing."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if force_regenerate or not os.path.exists(DATA_FILE):
        df = generate_clinical_cohort(n_samples=1200, random_state=42)
        df.to_csv(DATA_FILE, index=False)
    else:
        df = pd.read_csv(DATA_FILE)
    return df

if __name__ == "__main__":
    df = get_clinical_data(force_regenerate=True)
    print(f"Dataset generated at {DATA_FILE} with shape: {df.shape}")
    print("Class distribution:")
    print(df["heart_disease"].value_counts(normalize=True))
    print("\nGender distribution:")
    print(df["gender_label"].value_counts(normalize=True))
    print("\nPrevalence by Gender:")
    print(df.groupby("gender_label")["heart_disease"].mean())
