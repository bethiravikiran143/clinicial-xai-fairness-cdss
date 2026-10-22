"""
Model-Agnostic Explainable AI (XAI) Engine
Provides SHAP and LIME interpretations for individual patient predictions
and global black-box model auditing.
"""

import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer

FEATURE_LABELS = {
    "age": "Age (years)",
    "sex": "Biological Sex (0=F, 1=M)",
    "chest_pain_type": "Chest Pain Type (0:Typ, 1:Atyp, 2:Non-ang, 3:Asymp)",
    "resting_bp": "Resting Blood Pressure (mm Hg)",
    "cholesterol": "Serum Cholesterol (mg/dl)",
    "fasting_bs": "Fasting Blood Sugar > 120 (0=No, 1=Yes)",
    "resting_ecg": "Resting ECG (0:Norm, 1:ST-T, 2:LVH)",
    "max_heart_rate": "Max Heart Rate Achieved (bpm)",
    "exercise_angina": "Exercise-Induced Angina (0=No, 1=Yes)",
    "st_depression": "ST Depression (mm)",
    "st_slope": "ST Slope (0:Up, 1:Flat, 2:Down)",
    "num_major_vessels": "Major Vessels Colored (0-3)",
    "thalassemia": "Thalassemia (1:Norm, 2:Fixed, 3:Reversible)"
}

class ClinicalXAI:
    def __init__(self, model, X_train, feature_names=None):
        self.model = model
        self.X_train = np.array(X_train)
        self.feature_names = feature_names if feature_names else list(X_train.columns)
        
        # Initialize SHAP explainer
        # For tree ensembles we use TreeExplainer for efficiency, or default to general Explainer
        try:
            self.shap_explainer = shap.TreeExplainer(self.model)
        except Exception:
            self.shap_explainer = shap.Explainer(self.model.predict_proba, self.X_train[:100])

        # Initialize LIME Tabular Explainer
        self.lime_explainer = LimeTabularExplainer(
            training_data=self.X_train,
            feature_names=self.feature_names,
            class_names=["Low Risk (Healthy)", "High Risk (Heart Disease)"],
            mode="classification",
            random_state=42
        )

    def explain_patient_shap(self, patient_series_or_array):
        """
        Computes SHAP values for a single patient instance.
        Returns a DataFrame of features, patient values, and SHAP contributions.
        """
        arr = np.array(patient_series_or_array).reshape(1, -1)
        raw_shap = self.shap_explainer(arr)
        
        # Handle binary classification shap dimension differences
        if len(raw_shap.values.shape) == 3:
            # Shape is (1, num_features, 2)
            shap_vals = raw_shap.values[0, :, 1]
            base_val = raw_shap.base_values[0, 1] if np.ndim(raw_shap.base_values) > 1 else raw_shap.base_values[0]
        else:
            shap_vals = raw_shap.values[0]
            base_val = raw_shap.base_values[0] if np.ndim(raw_shap.base_values) > 0 else raw_shap.base_values

        df_explanation = pd.DataFrame({
            "Feature": self.feature_names,
            "Clinical Label": [FEATURE_LABELS.get(f, f) for f in self.feature_names],
            "Patient Value": arr[0],
            "SHAP Value": shap_vals,
            "Impact": np.where(shap_vals > 0, "Increases Risk", "Decreases Risk")
        }).sort_values(by="SHAP Value", key=abs, ascending=False)

        return df_explanation, float(base_val)

    def explain_patient_lime(self, patient_series_or_array, num_features=6):
        """
        Generates local surrogate decision rule explanation via LIME.
        """
        arr = np.array(patient_series_or_array).flatten()
        exp = self.lime_explainer.explain_instance(
            data_row=arr,
            predict_fn=self.model.predict_proba,
            num_features=num_features
        )
        lime_list = exp.as_list()
        # Returns list of tuples: [('st_depression > 1.50', 0.24), ...]
        return lime_list

    def get_global_shap_importance(self, X_sample):
        """
        Calculates mean absolute SHAP values across a cohort sample.
        """
        raw_shap = self.shap_explainer(np.array(X_sample))
        if len(raw_shap.values.shape) == 3:
            vals = np.abs(raw_shap.values[:, :, 1]).mean(axis=0)
        else:
            vals = np.abs(raw_shap.values).mean(axis=0)

        df_global = pd.DataFrame({
            "Feature": self.feature_names,
            "Clinical Label": [FEATURE_LABELS.get(f, f) for f in self.feature_names],
            "Mean |SHAP|": vals
        }).sort_values(by="Mean |SHAP|", ascending=False)
        return df_global
