"""
Fairness Auditing Engine for Clinical Decision Support Systems (CDSS)
Audits algorithmic equity across protected demographic attributes (Sex, Age)
using Disparate Impact, Demographic Parity, and Equalized Odds metrics.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from fairlearn.metrics import (
    MetricFrame, selection_rate,
    demographic_parity_difference, demographic_parity_ratio,
    equalized_odds_difference, equalized_odds_ratio
)

def compute_group_metrics(y_true, y_pred):
    """Computes comprehensive diagnostic metrics for a specific subgroup cohort."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    n_total = len(y_true)
    if n_total == 0:
        return {}

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Sensitivity / Recall
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # Specificity
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # Fall-out
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0  # Miss rate
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # Precision
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    return {
        "Cohort Size (N)": int(n_total),
        "Actual Prevalence": float(round(np.mean(y_true), 4)),
        "Selection Rate (Pred Positive)": float(round(np.mean(y_pred), 4)),
        "Accuracy": float(round(accuracy_score(y_true, y_pred), 4)),
        "Sensitivity / TPR": float(round(tpr, 4)),
        "Specificity / TNR": float(round(tnr, 4)),
        "False Positive Rate (FPR)": float(round(fpr, 4)),
        "False Negative Rate (FNR)": float(round(fnr, 4)),
        "Precision / PPV": float(round(ppv, 4)),
        "Negative Pred Value (NPV)": float(round(npv, 4)),
        "F1 Score": float(round(f1_score(y_true, y_pred, zero_division=0), 4))
    }

class ClinicalFairnessAuditor:
    def __init__(self, y_true, y_pred, sensitive_features, priv_group, unpriv_group):
        """
        Args:
            y_true: Ground truth binary diagnoses (0/1)
            y_pred: Model predicted binary diagnoses (0/1)
            sensitive_features: Series/array of demographic labels (e.g., 'Male', 'Female')
            priv_group: Identifier for the privileged group (e.g., 'Male')
            unpriv_group: Identifier for the unprivileged group (e.g., 'Female')
        """
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.sensitive_features = np.array(sensitive_features)
        self.priv_group = priv_group
        self.unpriv_group = unpriv_group

        self.mask_priv = (self.sensitive_features == priv_group)
        self.mask_unpriv = (self.sensitive_features == unpriv_group)

    def audit_summary(self):
        """
        Executes formal fairness audit and evaluates compliance against
        the 80% (4/5ths) Disparate Impact rule and Equalized Odds tolerance.
        """
        priv_metrics = compute_group_metrics(self.y_true[self.mask_priv], self.y_pred[self.mask_priv])
        unpriv_metrics = compute_group_metrics(self.y_true[self.mask_unpriv], self.y_pred[self.mask_unpriv])

        # 1. Disparate Impact (Statistical Parity Ratio)
        sr_priv = priv_metrics["Selection Rate (Pred Positive)"]
        sr_unpriv = unpriv_metrics["Selection Rate (Pred Positive)"]
        
        disparate_impact = (sr_unpriv / sr_priv) if sr_priv > 0 else 1.0
        demographic_parity_diff = abs(sr_priv - sr_unpriv)

        # 2. Equalized Odds metrics (TPR and FPR parity)
        tpr_priv = priv_metrics["Sensitivity / TPR"]
        tpr_unpriv = unpriv_metrics["Sensitivity / TPR"]
        fpr_priv = priv_metrics["False Positive Rate (FPR)"]
        fpr_unpriv = unpriv_metrics["False Positive Rate (FPR)"]

        tpr_diff = abs(tpr_priv - tpr_unpriv)
        fpr_diff = abs(fpr_priv - fpr_unpriv)
        equalized_odds_diff = max(tpr_diff, fpr_diff)
        equal_opportunity_diff = tpr_diff

        # 3. Compliance and Clinical Risk Evaluation
        # EEOC four-fifths benchmark: 0.80 <= Disparate Impact <= 1.25
        di_compliant = (disparate_impact >= 0.80) and (disparate_impact <= 1.25)
        # Clinical Equalized Odds tolerance is typically <= 0.10 (10% difference)
        eo_compliant = equalized_odds_diff <= 0.10

        audit_results = {
            "privileged_group": self.priv_group,
            "unprivileged_group": self.unpriv_group,
            "disparate_impact": round(disparate_impact, 4),
            "di_compliant": di_compliant,
            "demographic_parity_diff": round(demographic_parity_diff, 4),
            "equalized_odds_diff": round(equalized_odds_diff, 4),
            "eo_compliant": eo_compliant,
            "tpr_disparity": round(tpr_diff, 4),
            "fpr_disparity": round(fpr_diff, 4),
            "equal_opportunity_diff": round(equal_opportunity_diff, 4),
            "subgroups": {
                str(self.priv_group): priv_metrics,
                str(self.unpriv_group): unpriv_metrics
            }
        }

        # Clinical Diagnosis Implications
        clinical_notes = []
        if not di_compliant:
            clinical_notes.append(
                f"Disparate Impact ratio is {disparate_impact:.3f} (outside [0.80, 1.25] threshold). "
                f"Cohort '{self.unpriv_group}' receives positive diagnosis recommendations at a significantly different rate than '{self.priv_group}'."
            )
        else:
            clinical_notes.append(
                f"Disparate Impact ratio is {disparate_impact:.3f}, satisfying the 80% Statistical Parity standard."
            )

        if tpr_diff > 0.08:
            underdiagnosed = self.unpriv_group if (tpr_unpriv < tpr_priv) else self.priv_group
            clinical_notes.append(
                f"Sensitivity Disparity: Noticeable gap in True Positive Rate ({tpr_diff:.1%}). "
                f"Cohort '{underdiagnosed}' has higher risk of under-diagnosis / false negatives."
            )
        
        if fpr_diff > 0.08:
            overdiagnosed = self.unpriv_group if (fpr_unpriv > fpr_priv) else self.priv_group
            clinical_notes.append(
                f"Specificity Disparity: Noticeable gap in False Positive Rate ({fpr_diff:.1%}). "
                f"Cohort '{overdiagnosed}' may be subjected to excess invasive confirmatory procedures."
            )

        audit_results["clinical_interpretation"] = clinical_notes
        return audit_results
