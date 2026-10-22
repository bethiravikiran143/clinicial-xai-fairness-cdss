"""
CardioGuard CDSS: Clinical Decision Support System with Model-Agnostic XAI and Fairness Auditing
Explainable AI for Cardiovascular Diagnostics & Algorithmic Equity Auditing
"""

import os
import joblib
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.xai_engine import ClinicalXAI, FEATURE_LABELS
from src.fairness_engine import ClinicalFairnessAuditor
from src.train import train_and_evaluate

# Streamlit Page Configuration
st.set_page_config(
    page_title="CardioGuard CDSS | XAI & Fairness Auditing",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clinical styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.0rem;
        color: #4b5563;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #2563eb;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .patient-emr-card {
        background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%);
        color: white;
        padding: 18px 24px;
        border-radius: 12px;
        margin-bottom: 24px;
        box-shadow: 0 4px 10px rgba(30, 58, 138, 0.2);
    }
</style>
""", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "models")

@st.cache_resource
def load_all_artifacts():
    required_files = [
        "test_data_bundle.pkl", "xgb_model.pkl", "rf_model.pkl",
        "lr_model.pkl", "scaler.pkl", "metrics_summary.json"
    ]
    missing = any(not os.path.exists(os.path.join(MODELS_DIR, f)) for f in required_files)
    if missing:
        train_and_evaluate()

    try:
        bundle = joblib.load(os.path.join(MODELS_DIR, "test_data_bundle.pkl"))
        xgb_model = joblib.load(os.path.join(MODELS_DIR, "xgb_model.pkl"))
        rf_model = joblib.load(os.path.join(MODELS_DIR, "rf_model.pkl"))
        lr_model = joblib.load(os.path.join(MODELS_DIR, "lr_model.pkl"))
        scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
        with open(os.path.join(MODELS_DIR, "metrics_summary.json"), "r") as f:
            metrics_summary = json.load(f)
    except Exception:
        # Re-train in current environment if pickled files differ
        train_and_evaluate()
        bundle = joblib.load(os.path.join(MODELS_DIR, "test_data_bundle.pkl"))
        xgb_model = joblib.load(os.path.join(MODELS_DIR, "xgb_model.pkl"))
        rf_model = joblib.load(os.path.join(MODELS_DIR, "rf_model.pkl"))
        lr_model = joblib.load(os.path.join(MODELS_DIR, "lr_model.pkl"))
        scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
        with open(os.path.join(MODELS_DIR, "metrics_summary.json"), "r") as f:
            metrics_summary = json.load(f)

    # Initialize Explainers
    xai_xgb = ClinicalXAI(xgb_model, bundle["X_train"], bundle["feature_cols"])
    xai_rf = ClinicalXAI(rf_model, bundle["X_train"], bundle["feature_cols"])

    return {
        "bundle": bundle,
        "models": {
            "XGBoost (Black-box Gradient Boosted Trees)": (xgb_model, xai_xgb, False),
            "Random Forest (Black-box Ensemble)": (rf_model, xai_rf, False),
            "Logistic Regression (Linear Baseline)": (lr_model, None, True)
        },
        "scaler": scaler,
        "metrics_summary": metrics_summary
    }

artifacts = load_all_artifacts()
bundle = artifacts["bundle"]
X_test = bundle["X_test"]
X_test_scaled = bundle["X_test_scaled"]
y_test = bundle["y_test"]
sens_test = bundle["sens_test"]
feature_cols = bundle["feature_cols"]

# ==========================================
# SIDEBAR CONTROLS
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/heart-with-pulse.png", width=64)
    st.markdown("### **CDSS Controls**")
    
    selected_model_name = st.selectbox(
        "Classifier Architecture",
        list(artifacts["models"].keys())
    )
    active_model, active_xai, is_scaled = artifacts["models"][selected_model_name]

    decision_threshold = st.slider(
        "Diagnostic Decision Threshold",
        min_value=0.20,
        max_value=0.80,
        value=0.50,
        step=0.05,
        help="Probability cutoff for diagnosing Heart Disease positive."
    )

    st.markdown("---")
    st.markdown("### **Quick Clinical Presets**")
    preset_choice = st.selectbox(
        "Load Clinical Profile",
        [
            "Custom Patient Entry",
            "Preset 1: Robert Chen (High-Risk Male, 4th Check-up)",
            "Preset 2: Sarah Jenkins (Borderline Female, 2nd Check-up)",
            "Preset 3: Alex Rivera (Healthy Adult, 1st Check-up)"
        ]
    )

    preset_values = {
        "Preset 1: Robert Chen (High-Risk Male, 4th Check-up)": {
            "patient_name": "Robert Chen",
            "patient_id": "PT-89104",
            "num_checkups": 4,
            "attending_doc": "Dr. Sarah Adams, MD",
            "chief_complaint": "Recurrent exertional retrosternal pressure, chronic hypertension",
            "age": 67, "sex": 1, "cp": 3, "trestbps": 162, "chol": 286,
            "fbs": 1, "restecg": 1, "thalach": 118, "exang": 1, "oldpeak": 2.8,
            "slope": 1, "ca": 2, "thal": 3
        },
        "Preset 2: Sarah Jenkins (Borderline Female, 2nd Check-up)": {
            "patient_name": "Sarah Jenkins",
            "patient_id": "PT-41092",
            "num_checkups": 2,
            "attending_doc": "Dr. Marcus Lee, MD",
            "chief_complaint": "Atypical epigastric tightness, exertional dyspnea & fatigue",
            "age": 58, "sex": 0, "cp": 1, "trestbps": 138, "chol": 242,
            "fbs": 0, "restecg": 0, "thalach": 145, "exang": 0, "oldpeak": 1.2,
            "slope": 0, "ca": 0, "thal": 2
        },
        "Preset 3: Alex Rivera (Healthy Adult, 1st Check-up)": {
            "patient_name": "Alex Rivera",
            "patient_id": "PT-11409",
            "num_checkups": 1,
            "attending_doc": "Dr. Michael Vance, MD",
            "chief_complaint": "Annual baseline executive health screening; asymptomatic",
            "age": 36, "sex": 1, "cp": 0, "trestbps": 118, "chol": 178,
            "fbs": 0, "restecg": 0, "thalach": 174, "exang": 0, "oldpeak": 0.0,
            "slope": 0, "ca": 0, "thal": 1
        }
    }

# ==========================================
# MAIN HEADER
# ==========================================
st.markdown('<div class="main-title">🩺 CardioGuard CDSS: XAI & Fairness Auditing</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Model-Agnostic Interpretability for Black-Box Classifiers via Fairness Auditing (Disparate Impact & Equalized Odds)</div>', unsafe_allow_html=True)

# Top KPI ribbon for Model Performance
m_info = artifacts["metrics_summary"].get(selected_model_name.split(" ")[0], {})
kpi_cols = st.columns(6)
kpi_cols[0].metric("Accuracy", f"{m_info.get('Accuracy', 0.0):.1%}")
kpi_cols[1].metric("Sensitivity (Recall)", f"{m_info.get('Sensitivity (Recall)', 0.0):.1%}")
kpi_cols[2].metric("Specificity", f"{m_info.get('Specificity', 0.0):.1%}")
kpi_cols[3].metric("Precision", f"{m_info.get('Precision', 0.0):.1%}")
kpi_cols[4].metric("F1-Score", f"{m_info.get('F1 Score', 0.0):.3f}")
kpi_cols[5].metric("ROC-AUC", f"{m_info.get('ROC-AUC', 0.0):.3f}")

# TABS (Exactly the 3 operational clinical tabs)
tab_diag, tab_global, tab_fairness = st.tabs([
    "🔍 Patient Diagnosis & Explanation (Local XAI)",
    "🌐 Global Black-Box Interpretability",
    "⚖️ Algorithmic Fairness Audit"
])

# ==========================================
# TAB 1: LOCAL PATIENT DIAGNOSIS & EXPLANATION
# ==========================================
with tab_diag:
    # Defaults from preset
    pv = preset_values.get(preset_choice, {
        "patient_name": "Jane Doe",
        "patient_id": "PT-10254",
        "num_checkups": 1,
        "attending_doc": "Dr. Sarah Adams, MD",
        "chief_complaint": "Routine cardiovascular consultation and evaluation",
        "age": 55, "sex": 1, "cp": 2, "trestbps": 130, "chol": 240,
        "fbs": 0, "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 0.8,
        "slope": 1, "ca": 0, "thal": 2
    })

    st.markdown("#### **1. Patient Demographics & Electronic Medical Record (EMR)**")
    p_col1, p_col2, p_col3 = st.columns([1.2, 1, 1.8])
    with p_col1:
        in_name = st.text_input("Patient Full Name", value=pv.get("patient_name", "Jane Doe"))
        in_mrn = st.text_input("Medical Record Number (MRN / Patient ID)", value=pv.get("patient_id", "PT-10254"))
    with p_col2:
        in_checkups = st.number_input(
            "Check-up / Visit Number", 
            min_value=1, max_value=50, 
            value=int(pv.get("num_checkups", 1)),
            help="Cumulative number of clinical check-ups for this patient."
        )
        in_doctor = st.text_input("Attending Physician", value=pv.get("attending_doc", "Dr. Sarah Adams, MD"))
    with p_col3:
        in_complaint = st.text_area(
            "Chief Complaint & Clinical Notes",
            value=pv.get("chief_complaint", "Routine cardiology follow-up"),
            height=100
        )

    st.markdown("---")
    st.markdown("#### **2. Patient Physiological & Diagnostic Vitals**")
    
    # Human-readable mappings for clean UI display
    sex_map = {0: "Female", 1: "Male"}
    cp_map = {0: "Typical Angina (0)", 1: "Atypical Angina (1)", 2: "Non-anginal Pain (2)", 3: "Asymptomatic (3)"}
    fbs_map = {0: "Normal (<= 120 mg/dl)", 1: "Elevated (> 120 mg/dl)"}
    restecg_map = {0: "Normal (0)", 1: "ST-T Wave Abnormality (1)", 2: "Left Ventricular Hypertrophy (2)"}
    exang_map = {0: "No (0)", 1: "Yes (1)"}
    slope_map = {0: "Upsloping (0)", 1: "Flat (1)", 2: "Downsloping (2)"}
    thal_map = {1: "Normal (1)", 2: "Fixed Defect (2)", 3: "Reversible Defect (3)"}

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        in_age = st.number_input("Age (Years)", 20, 95, int(pv["age"]))
        in_sex = st.selectbox("Biological Sex", options=[0, 1], format_func=lambda x: sex_map[x], index=int(pv["sex"]))
        in_cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3], format_func=lambda x: cp_map[x], index=int(pv["cp"]))
        in_trestbps = st.number_input("Resting Blood Pressure (mmHg)", 80, 220, int(pv["trestbps"]))

    with c2:
        in_chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, int(pv["chol"]))
        in_fbs = st.selectbox("Fasting Blood Sugar", options=[0, 1], format_func=lambda x: fbs_map[x], index=int(pv["fbs"]))
        in_restecg = st.selectbox("Resting ECG Results", options=[0, 1, 2], format_func=lambda x: restecg_map[x], index=int(pv["restecg"]))
        in_thalach = st.number_input("Max Heart Rate Achieved (bpm)", 60, 220, int(pv["thalach"]))

    with c3:
        in_exang = st.selectbox("Exercise-Induced Angina", options=[0, 1], format_func=lambda x: exang_map[x], index=int(pv["exang"]))
        in_oldpeak = st.number_input("ST Depression (mm)", 0.0, 7.0, float(pv["oldpeak"]), step=0.1)
        in_slope = st.selectbox("Slope of Peak ST", options=[0, 1, 2], format_func=lambda x: slope_map[x], index=int(pv["slope"]))

    with c4:
        in_ca = st.selectbox("Major Vessels Colored (Fluoroscopy)", options=[0, 1, 2, 3], index=int(pv["ca"]))
        in_thal = st.selectbox("Thalassemia Type", options=[1, 2, 3], format_func=lambda x: thal_map[x], index=int(pv["thal"])-1)
        st.markdown("<br>", unsafe_allow_html=True)
        btn_diagnose = st.button("🩺 Run CDSS Diagnostic Inference", use_container_width=True, type="primary")

    # Patient row vector
    patient_data = pd.DataFrame([{
        "age": in_age, "sex": in_sex, "chest_pain_type": in_cp, "resting_bp": in_trestbps,
        "cholesterol": in_chol, "fasting_bs": in_fbs, "resting_ecg": in_restecg,
        "max_heart_rate": in_thalach, "exercise_angina": in_exang, "st_depression": in_oldpeak,
        "st_slope": in_slope, "num_major_vessels": in_ca, "thalassemia": in_thal
    }])

    # Model inference
    if is_scaled:
        eval_patient = artifacts["scaler"].transform(patient_data)
    else:
        eval_patient = patient_data

    prob_disease = float(active_model.predict_proba(eval_patient)[0, 1])
    is_positive = prob_disease >= decision_threshold

    st.markdown("---")
    st.markdown("#### **3. Clinical Diagnostic Assessment & Patient Chart**")
    
    # Patient EMR Chart Summary Card
    st.markdown(f"""
    <div class="patient-emr-card">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap;">
            <div>
                <span style="background-color: rgba(255,255,255,0.25); padding: 4px 10px; border-radius: 6px; font-size: 0.8rem; font-weight: 700; letter-spacing: 0.8px;">CLINICAL EMR RECORD</span>
                <h3 style="margin: 8px 0 4px 0; color: white; font-size: 1.6rem;">👤 {in_name}</h3>
                <span style="opacity: 0.95; font-size: 0.95rem;">
                    <b>MRN:</b> <code>{in_mrn}</code> &nbsp;|&nbsp; 
                    <b>Age / Sex:</b> {in_age} yrs / {sex_map[in_sex]} &nbsp;|&nbsp; 
                    <b>Consultation:</b> Check-up #{in_checkups}
                </span>
            </div>
            <div style="text-align: right; font-size: 0.92rem; max-width: 480px;">
                <div style="margin-bottom: 4px;">👨‍⚕️ <b>Attending:</b> {in_doctor}</div>
                <div style="opacity: 0.9; font-style: italic;"><b>Notes:</b> {in_complaint}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    r1, r2 = st.columns([1, 2])
    with r1:
        # Gauge chart for risk score
        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob_disease * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Cardiovascular Disease Risk", 'font': {'size': 18}},
            number={'suffix': "%", 'font': {'size': 28}},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#dc2626" if is_positive else "#16a34a"},
                'steps': [
                    {'range': [0, decision_threshold * 100], 'color': "#dcfce7"},
                    {'range': [decision_threshold * 100, 100], 'color': "#fee2e2"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 3},
                    'thickness': 0.75,
                    'value': decision_threshold * 100
                }
            }
        ))
        gauge_fig.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(gauge_fig, use_container_width=True)

    with r2:
        if is_positive:
            st.error(f"""
            ### ⚠️ HIGH RISK DETECTED ({prob_disease:.1%}) for {in_name}
            **Diagnostic Recommendation:** Calculated risk exceeds decision threshold ({decision_threshold:.0%}).
            Immediate referral to invasive cardiology recommended for coronary angiography, stress echocardiography, and intensive pharmacotherapy.
            """)
        else:
            st.success(f"""
            ### ✅ LOW RISK / HEALTHY ({prob_disease:.1%}) for {in_name}
            **Diagnostic Recommendation:** Calculated risk falls below decision threshold ({decision_threshold:.0%}).
            Continue routine outpatient cardiovascular follow-up, dietary lifestyle modifications, and monitor at next scheduled check-up (#{in_checkups + 1}).
            """)
        
        st.info(f"""
        **Patient Context:** {in_name} (MRN: `{in_mrn}`), Check-up #{in_checkups}.
        **Demographic Profile:** {sex_map[in_sex]}, Age {in_age} ({'Senior' if in_age >= 55 else 'Adult'}).
        **Decision Rule:** P(Disease) $\\ge$ {decision_threshold:.2f} $\\implies$ High Risk.
        """)

    # Explainability Section
    st.markdown("---")
    st.markdown("#### **4. Model-Agnostic Explainability (SHAP & LIME)**")
    
    if active_xai is None:
        st.warning("Model-agnostic SHAP & LIME explanations are designed for the non-linear Black-box models (XGBoost & Random Forest). Please select XGBoost or Random Forest from the sidebar.")
    else:
        xai_tab1, xai_tab2, xai_tab3 = st.tabs([
            "📊 SHAP Patient Attribution",
            "🧩 LIME Local Decision Rules",
            "📝 Automated Clinical Narrative"
        ])

        with xai_tab1:
            st.markdown(f"##### **SHAP Attribution for {in_name}**")
            st.caption("Shows how each physiological metric pushes the patient's risk higher (Red) or lower (Green) relative to the baseline population.")
            
            df_shap, base_val = active_xai.explain_patient_shap(patient_data.iloc[0])
            
            shap_plot_df = df_shap.head(10).copy()
            fig_shap = px.bar(
                shap_plot_df,
                x="SHAP Value",
                y="Clinical Label",
                orientation="h",
                color="Impact",
                color_discrete_map={"Increases Risk": "#ef4444", "Decreases Risk": "#10b981"},
                title=f"Top Clinical Features Influencing Diagnosis for {in_name}",
                hover_data=["Patient Value"]
            )
            fig_shap.update_layout(yaxis={'categoryorder': 'total ascending'}, height=400)
            st.plotly_chart(fig_shap, use_container_width=True)

            with st.expander("View Full SHAP Metric Breakdown Table"):
                st.dataframe(df_shap[["Feature", "Clinical Label", "Patient Value", "SHAP Value", "Impact"]], use_container_width=True)

        with xai_tab2:
            st.markdown(f"##### **LIME Local Decision Rules for {in_name}**")
            st.caption("Surrogate local linear approximation identifying explicit clinical threshold rules driving this prediction.")
            
            lime_rules = active_xai.explain_patient_lime(patient_data.iloc[0], num_features=6)
            lime_df = pd.DataFrame(lime_rules, columns=["Clinical Rule Condition", "Local Weight"])
            lime_df["Direction"] = np.where(lime_df["Local Weight"] > 0, "Increases Risk", "Decreases Risk")

            fig_lime = px.bar(
                lime_df,
                x="Local Weight",
                y="Clinical Rule Condition",
                orientation="h",
                color="Direction",
                color_discrete_map={"Increases Risk": "#ef4444", "Decreases Risk": "#10b981"},
                title=f"LIME Decision Bounds for {in_name}"
            )
            fig_lime.update_layout(yaxis={'categoryorder': 'total ascending'}, height=350)
            st.plotly_chart(fig_lime, use_container_width=True)

        with xai_tab3:
            st.markdown(f"##### **Automated Physician Explanation Note for {in_name}**")
            top_risk_factors = df_shap[df_shap["SHAP Value"] > 0].head(3)
            top_protective_factors = df_shap[df_shap["SHAP Value"] < 0].head(3)
            
            risk_bullets = "".join([f"- **{row['Clinical Label']}**: Recorded value of `{row['Patient Value']}` (SHAP contribution: +{row['SHAP Value']:.3f})\n" for _, row in top_risk_factors.iterrows()])
            prot_bullets = "".join([f"- **{row['Clinical Label']}**: Recorded value of `{row['Patient Value']}` (SHAP contribution: {row['SHAP Value']:.3f})\n" for _, row in top_protective_factors.iterrows()])

            st.markdown(f"""
            **Clinical Decision Explanation Summary:**
            * **Patient:** {in_name} (MRN: `{in_mrn}`)
            * **Consultation:** Check-up #{in_checkups} | **Attending:** {in_doctor}
            * **Clinical Presentation:** {in_complaint}
            * **Assigned Cardiovascular Disease Probability:** **{prob_disease:.1%}** ({'HIGH RISK' if is_positive else 'LOW RISK'}).
            
            **Key Pathological Drivers (Elevating Risk):**
            {risk_bullets if risk_bullets else "- No significant risk factors elevated above baseline."}

            **Key Protective Factors (Mitigating Risk):**
            {prot_bullets if prot_bullets else "- No significant protective factors detected."}

            *Note: This explanation is generated via model-agnostic Shapley attribution to provide transparency into black-box clinical classifications.*
            """)

# ==========================================
# TAB 2: GLOBAL MODEL INTERPRETABILITY
# ==========================================
with tab_global:
    st.markdown("#### **Global Black-box Feature Importance Across Validation Cohort**")
    st.caption("Calculates the global mean absolute SHAP value across the clinical cohort ($N = 300$).")

    if active_xai is not None:
        df_global = active_xai.get_global_shap_importance(X_test)
        
        fig_global = px.bar(
            df_global,
            x="Mean |SHAP|",
            y="Clinical Label",
            orientation="h",
            title=f"Global Feature Importance Ranking ({selected_model_name.split(' ')[0]})",
            color="Mean |SHAP|",
            color_continuous_scale="Blues"
        )
        fig_global.update_layout(yaxis={'categoryorder': 'total ascending'}, height=450)
        st.plotly_chart(fig_global, use_container_width=True)

        st.markdown("##### **Clinical Risk Factor Distributions**")
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            fig_hist1 = px.histogram(
                X_test, x="st_depression", color=y_test.map({0: "Healthy", 1: "Heart Disease"}),
                title="ST Depression vs Ground Truth Diagnosis", barmode="overlay", nbins=20
            )
            st.plotly_chart(fig_hist1, use_container_width=True)
        with col_g2:
            fig_hist2 = px.histogram(
                X_test, x="max_heart_rate", color=y_test.map({0: "Healthy", 1: "Heart Disease"}),
                title="Max Heart Rate Achieved vs Ground Truth Diagnosis", barmode="overlay", nbins=20
            )
            st.plotly_chart(fig_hist2, use_container_width=True)
    else:
        st.info("Global SHAP analysis available when XGBoost or Random Forest is selected.")

# ==========================================
# TAB 3: FAIRNESS AUDITING
# ==========================================
with tab_fairness:
    st.markdown("#### **Algorithmic Fairness & Demographic Auditing**")
    st.markdown("Audits whether the black-box classifier demonstrates demographic disparity or discriminatory diagnosis rates.")

    f_col1, f_col2 = st.columns([1, 2])
    with f_col1:
        audit_attr = st.selectbox(
            "Protected Demographic Attribute",
            ["Biological Sex (Gender)", "Age Group (<55 vs >=55)"]
        )

    if audit_attr == "Biological Sex (Gender)":
        sens_column = sens_test["gender_label"]
        priv_name = "Male"
        unpriv_name = "Female"
    else:
        sens_column = sens_test["age_group"]
        priv_name = "Senior (>=55)"
        unpriv_name = "Adult (<55)"

    # Compute test set predictions using the chosen threshold
    if is_scaled:
        test_probs = active_model.predict_proba(X_test_scaled)[:, 1]
    else:
        test_probs = active_model.predict_proba(X_test)[:, 1]
    
    test_preds = (test_probs >= decision_threshold).astype(int)

    auditor = ClinicalFairnessAuditor(
        y_true=y_test,
        y_pred=test_preds,
        sensitive_features=sens_column,
        priv_group=priv_name,
        unpriv_group=unpriv_name
    )
    audit = auditor.audit_summary()

    # Metric Scorecards
    st.markdown("##### **1. Primary Fairness Audit Scorecards**")
    sc1, sc2, sc3, sc4 = st.columns(4)

    di_val = audit["disparate_impact"]
    di_badge = "✅ COMPLIANT" if audit["di_compliant"] else "❌ VIOLATION"
    sc1.metric(
        "Disparate Impact (Statistical Parity)",
        f"{di_val:.3f}",
        delta=di_badge,
        delta_color="normal" if audit["di_compliant"] else "inverse"
    )

    eo_val = audit["equalized_odds_diff"]
    eo_badge = "✅ BALANCED" if audit["eo_compliant"] else "⚠️ DISPARITY"
    sc2.metric(
        "Equalized Odds Difference",
        f"{eo_val:.3f}",
        delta=eo_badge,
        delta_color="normal" if audit["eo_compliant"] else "inverse"
    )

    tpr_disp = audit["tpr_disparity"]
    sc3.metric("Equal Opportunity (TPR Diff)", f"{tpr_disp:.3f}", delta=f"{tpr_disp*100:.1f}% Gap")

    fpr_disp = audit["fpr_disparity"]
    sc4.metric("False Positive Parity (FPR Diff)", f"{fpr_disp:.3f}", delta=f"{fpr_disp*100:.1f}% Gap")

    st.markdown("##### **2. Demographic Subgroup Performance Breakdown**")
    subgroups = audit["subgroups"]
    sub_df = pd.DataFrame(subgroups).T
    st.dataframe(sub_df.style.format({
        "Actual Prevalence": "{:.2%}",
        "Selection Rate (Pred Positive)": "{:.2%}",
        "Accuracy": "{:.2%}",
        "Sensitivity / TPR": "{:.2%}",
        "Specificity / TNR": "{:.2%}",
        "False Positive Rate (FPR)": "{:.2%}",
        "False Negative Rate (FNR)": "{:.2%}",
        "Precision / PPV": "{:.2%}",
        "Negative Pred Value (NPV)": "{:.2%}",
        "F1 Score": "{:.3f}"
    }), use_container_width=True)

    # Subgroup Comparison Bar Chart
    comp_df = pd.DataFrame({
        "Metric": ["Selection Rate", "Sensitivity (TPR)", "Specificity (TNR)", "False Positive Rate (FPR)"],
        priv_name: [
            subgroups[priv_name]["Selection Rate (Pred Positive)"],
            subgroups[priv_name]["Sensitivity / TPR"],
            subgroups[priv_name]["Specificity / TNR"],
            subgroups[priv_name]["False Positive Rate (FPR)"]
        ],
        unpriv_name: [
            subgroups[unpriv_name]["Selection Rate (Pred Positive)"],
            subgroups[unpriv_name]["Sensitivity / TPR"],
            subgroups[unpriv_name]["Specificity / TNR"],
            subgroups[unpriv_name]["False Positive Rate (FPR)"]
        ]
    }).melt(id_vars="Metric", var_name="Demographic Cohort", value_name="Score")

    fig_comp = px.bar(
        comp_df,
        x="Metric",
        y="Score",
        color="Demographic Cohort",
        barmode="group",
        title=f"Diagnostic Equity Comparison: {priv_name} vs {unpriv_name}",
        color_discrete_sequence=["#2563eb", "#ec4899"]
    )
    st.plotly_chart(fig_comp, use_container_width=True)

    st.markdown("##### **3. Clinical & Ethical Audit Findings**")
    for note in audit["clinical_interpretation"]:
        st.warning(note)
