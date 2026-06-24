import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ----------------------------------------------------------------------------
# Load trained model, scaler, and metadata (produced by train_model_v3.py)
# ----------------------------------------------------------------------------
@st.cache_resource
def load_resources():
    required_files = [
        "best_model.pkl",
        "scaler.pkl",
        "model_metadata.pkl"
    ]

    missing = [f for f in required_files if not Path(f).exists()]

    if missing:
        raise FileNotFoundError(
            f"Missing required files: {', '.join(missing)}"
        )

    model = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")
    meta = joblib.load("model_metadata.pkl")

    required_keys = [
        "feature_columns",
        "numeric_features",
        "best_model_uses_decision_function",
        "best_model_name",
        "test_macro_f1",
        "feature_importances",
    ]

    for key in required_keys:
        if key not in meta:
            raise KeyError(
                f"Metadata missing required key: {key}"
            )

    return model, scaler, meta


try:
    model, scaler, meta = load_resources()
    feature_columns = meta["feature_columns"]
    numeric_features = meta["numeric_features"]
    uses_decision_function = meta["best_model_uses_decision_function"]

except Exception as e:
    st.set_page_config(
        page_title="Hypertension Risk Estimator",
        page_icon="🩺"
    )

    st.error(
        f"Application startup failed.\n\n{str(e)}"
    )

    st.stop()
st.set_page_config(page_title="Hypertension Risk Estimator", page_icon="🩺")

st.title("🩺 Hypertension (High Blood Pressure) Risk Estimator")

st.markdown(
    f"""
This tool estimates your likelihood of having high blood pressure (hypertension)
based on lifestyle, vitals, and general health indicators, using a
**{meta['best_model_name']}** model trained on real BRFSS 2015 health survey data
(70,692 respondents).

This is **not a medical diagnosis**. It's a statistical estimate for awareness
and self-reflection only. Please consult a healthcare provider for an actual
diagnosis.
"""
)

user_input = {}

tabs = st.tabs(
    ["🧍 Profile & Vitals", "🥗 Lifestyle", "🩺 Health History", "📋 General Health"]
)

with tabs[0]:
    age_bucket = st.selectbox(
        "Age group",
        options=list(range(1, 14)),
        format_func=lambda x: {
            1: "18-24",
            2: "25-29",
            3: "30-34",
            4: "35-39",
            5: "40-44",
            6: "45-49",
            7: "50-54",
            8: "55-59",
            9: "60-64",
            10: "65-69",
            11: "70-74",
            12: "75-79",
            13: "80+",
        }[x],
        index=7,
    )

    user_input["Age"] = age_bucket
    user_input["Sex"] = (
        1 if st.radio("Sex", ["Female", "Male"]) == "Male" else 0
    )
    user_input["BMI"] = st.slider("BMI (kg/m²)", 12, 60, 27, step=1)

with tabs[1]:
    user_input["Smoker"] = (
        1 if st.checkbox("I have smoked at least 100 cigarettes in my life") else 0
    )

    user_input["HvyAlcoholConsump"] = (
        1
        if st.checkbox(
            "Heavy alcohol consumption (14+ drinks/week men, 7+ women)"
        )
        else 0
    )

    user_input["PhysActivity"] = (
        1
        if st.checkbox(
            "Physically active in the last 30 days (outside of job)"
        )
        else 0
    )

    user_input["Fruits"] = (
        1 if st.checkbox("Eat fruit 1+ times per day") else 0
    )

    user_input["Veggies"] = (
        1 if st.checkbox("Eat vegetables 1+ times per day") else 0
    )

with tabs[2]:
    user_input["HighChol"] = (
        1
        if st.checkbox(
            "Told by a doctor I have high cholesterol"
        )
        else 0
    )

    user_input["CholCheck"] = (
        1
        if st.checkbox(
            "Had a cholesterol check in the last 5 years"
        )
        else 0
    )

    user_input["Diabetes"] = (
        1
        if st.checkbox(
            "Diagnosed with diabetes (any type)"
        )
        else 0
    )

    user_input["HeartDiseaseorAttack"] = (
        1
        if st.checkbox(
            "History of coronary heart disease or heart attack"
        )
        else 0
    )

    user_input["Stroke"] = (
        1 if st.checkbox("History of stroke") else 0
    )

    user_input["DiffWalk"] = (
        1
        if st.checkbox(
            "Serious difficulty walking or climbing stairs"
        )
        else 0
    )

with tabs[3]:
    user_input["GenHlth"] = st.slider(
        "General health (1 = Excellent, 5 = Poor)",
        1,
        5,
        3,
    )

    user_input["MentHlth"] = st.slider(
        "Days in past 30 with poor mental health",
        0,
        30,
        0,
    )

    user_input["PhysHlth"] = st.slider(
        "Days in past 30 with poor physical health",
        0,
        30,
        0,
    )

# ----------------------------------------------------------------------------
# Predict
# ----------------------------------------------------------------------------
if st.button("🔎 Estimate My Risk"):

    try:

        row = pd.DataFrame([user_input])

        missing_features = [
            col
            for col in feature_columns
            if col not in row.columns
        ]

        if missing_features:
            st.error(
                f"Missing required inputs: {missing_features}"
            )
            st.stop()

        row = row[feature_columns]

        row_scaled = row.copy()

        row_scaled[numeric_features] = scaler.transform(
            row[numeric_features]
        )

        if uses_decision_function:

            raw_score = float(
                model.decision_function(row_scaled)[0]
            )

            raw_score = np.clip(
                raw_score,
                -500,
                500,
            )

            prob = 1 / (
                1 + np.exp(-raw_score)
            )

        else:

            prob = model.predict_proba(
                row_scaled
            )[0][1]

        prob = float(prob)

        if np.isnan(prob) or np.isinf(prob):
            raise ValueError(
                "Model returned an invalid probability."
            )

        prob = max(
            0.0,
            min(prob, 1.0)
        )

        risk_pct = float(
            round(prob * 100, 1)
        )

        st.subheader(
            f"🧮 Estimated Risk: {risk_pct}%"
        )

        st.progress(
            int(prob * 100)
        )

        if risk_pct < 35:

            st.success(
                "🟢 Lower likelihood of high blood pressure based on these factors."
            )

        elif risk_pct < 65:

            st.warning(
                "🟠 Moderate likelihood. Worth discussing with a healthcare provider."
            )

        else:

            st.error(
                "🔴 Higher likelihood of high blood pressure. Please consult a healthcare provider."
            )

        st.caption(
            f"Model: {meta['best_model_name']} | "
            f"Test-set macro F1: {meta['test_macro_f1']:.2f} "
            f"| Trained on real BRFSS 2015 survey data."
        )

        with st.expander(
            "📊 What drives this estimate?"
        ):

            try:

                importances = pd.Series(
                    meta["feature_importances"]
                ).sort_values(
                    ascending=False
                )

                st.bar_chart(
                    importances
                )

            except Exception as e:

                st.warning(
                    f"Could not display feature importances: {e}"
                )

        try:

            df_report = pd.DataFrame.from_dict(
                user_input,
                orient="index",
                columns=["Value"],
            )

            df_report.reset_index(
                inplace=True
            )

            df_report.rename(
                columns={
                    "index": "Feature"
                },
                inplace=True,
            )

            df_report.loc[
                len(df_report.index)
            ] = [
                "Estimated Risk (%)",
                risk_pct,
            ]

            csv = (
                df_report
                .to_csv(index=False)
                .encode("utf-8")
            )

            st.download_button(
                "⬇️ Download Your Report as CSV",
                data=csv,
                file_name="hypertension_risk_report.csv",
                mime="text/csv",
            )

        except Exception as e:

            st.warning(
                f"Report generation failed: {e}"
            )

    except Exception as e:

        st.error(
            "Prediction could not be completed."
        )

        with st.expander(
            "Technical details"
        ):
            st.code(str(e))