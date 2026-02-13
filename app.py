
# ============================================================
# app.py — Dry Bean Classification Streamlit App
# ML Assignment 2 | BITS Pilani M.Tech AIML/DSE
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

# ---- Page config ----
st.set_page_config(
    page_title="Dry Bean Classifier",
    page_icon="🫘",
    layout="wide"
)

# ---- Load saved artifacts ----
@st.cache_resource
def load_artifacts():
    with open("model/scaler.pkl",        "rb") as f: scaler    = pickle.load(f)
    with open("model/label_encoder.pkl", "rb") as f: le        = pickle.load(f)
    with open("model/feature_cols.pkl",  "rb") as f: feat_cols = pickle.load(f)

    models = {}
    model_files = {
        "Logistic Regression" : "model/logistic_regression.pkl",
        "Decision Tree"       : "model/decision_tree.pkl",
        "KNN"                 : "model/knn.pkl",
        "Naive Bayes"         : "model/naive_bayes.pkl",
        "Random Forest"       : "model/random_forest.pkl",
        "XGBoost"             : "model/xgboost.pkl"
    }
    for name, path in model_files.items():
        with open(path, "rb") as f:
            models[name] = pickle.load(f)

    metrics_df = pd.read_csv("model/metrics_results.csv", index_col=0)
    return scaler, le, feat_cols, models, metrics_df

scaler, le, feat_cols, models, metrics_df = load_artifacts()
class_names = list(le.classes_)
NEEDS_SCALING = {"Logistic Regression", "KNN", "Naive Bayes"}

# ---- Header ----
st.title("🫘 Dry Bean Classifier — ML Assignment 2")
st.markdown("""
**BITS Pilani | M.Tech AIML/DSE | Machine Learning**

This app classifies dry bean varieties using 6 trained ML models.
Upload a test CSV file, choose a model, and view predictions + metrics.
""")
st.divider()

# ---- Sidebar ----
with st.sidebar:
    st.header("⚙️ Controls")
    selected_model = st.selectbox(
        "Select Classification Model",
        list(models.keys())
    )
    st.info(f"Selected: **{selected_model}**")
    st.markdown("---")
    st.markdown("📥 **Upload test CSV below (main panel)**")
    st.markdown("The CSV must contain the 16 bean feature columns + Class column.")
    st.markdown("Use `test_data_sample.csv` from the repository.")

# ---- Tabs ----
tab1, tab2, tab3 = st.tabs(["📤 Upload & Predict", "📊 Metrics Comparison", "ℹ️ About Dataset"])

# ===========================
# TAB 1: Upload & Predict
# ===========================
with tab1:
    st.subheader("📤 Upload Test Data (CSV)")
    uploaded_file = st.file_uploader(
        "Upload CSV file with bean features + Class column",
        type=["csv"],
        help="Upload test_data_sample.csv from the repository."
    )

    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            st.success(f"File uploaded: {df_upload.shape[0]} rows × {df_upload.shape[1]} cols")
            st.dataframe(df_upload.head(5), use_container_width=True)

            # Check required columns
            missing_cols = [c for c in feat_cols if c not in df_upload.columns]
            has_target   = "Class" in df_upload.columns

            if missing_cols:
                st.error(f"Missing feature columns: {missing_cols}")
            else:
                X_up = df_upload[feat_cols].copy()
                model = models[selected_model]

                # Scale if needed
                if selected_model in NEEDS_SCALING:
                    X_input = scaler.transform(X_up)
                else:
                    X_input = X_up.values

                # Predict
                y_pred = model.predict(X_input)
                y_prob = model.predict_proba(X_input)
                pred_labels = le.inverse_transform(y_pred)

                st.subheader(f"🔮 Predictions — {selected_model}")
                result_df = df_upload[feat_cols].copy()
                result_df.insert(0, "Predicted Class", pred_labels)
                if has_target:
                    result_df.insert(1, "Actual Class", df_upload["Class"].values)
                    result_df.insert(2, "Correct",
                        ["✅" if p == a else "❌"
                         for p, a in zip(pred_labels, df_upload["Class"].values)])
                st.dataframe(result_df.head(20), use_container_width=True)

                if has_target:
                    st.subheader("📈 Evaluation Metrics")
                    y_true_enc = le.transform(df_upload["Class"].values)
                    n_cls = len(class_names)

                    acc  = accuracy_score(y_true_enc, y_pred)
                    prec = precision_score(y_true_enc, y_pred, average="weighted", zero_division=0)
                    rec  = recall_score(y_true_enc, y_pred, average="weighted", zero_division=0)
                    f1   = f1_score(y_true_enc, y_pred, average="weighted", zero_division=0)
                    mcc  = matthews_corrcoef(y_true_enc, y_pred)
                    try:
                        if n_cls == 2:
                            auc = roc_auc_score(y_true_enc, y_prob[:, 1])
                        else:
                            auc = roc_auc_score(y_true_enc, y_prob, multi_class="ovr", average="macro")
                    except Exception:
                        auc = float("nan")

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Accuracy",  f"{acc*100:.2f}%")
                    col1.metric("AUC Score", f"{auc*100:.2f}%" if not np.isnan(auc) else "N/A")
                    col2.metric("Precision", f"{prec*100:.2f}%")
                    col2.metric("Recall",    f"{rec*100:.2f}%")
                    col3.metric("F1 Score",  f"{f1*100:.2f}%")
                    col3.metric("MCC Score", f"{mcc:.4f}")

                    # Confusion Matrix
                    st.subheader("🔲 Confusion Matrix")
                    cm = confusion_matrix(y_true_enc, y_pred)
                    fig, ax = plt.subplots(figsize=(9, 6))
                    sns.heatmap(
                        cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=class_names, yticklabels=class_names,
                        linewidths=0.5, annot_kws={"size": 11}
                    )
                    ax.set_title(f"Confusion Matrix — {selected_model}",
                                 fontsize=13, fontweight="bold")
                    ax.set_xlabel("Predicted Label", fontsize=11)
                    ax.set_ylabel("True Label", fontsize=11)
                    ax.tick_params(axis="x", rotation=30)
                    plt.tight_layout()
                    st.pyplot(fig)

                    # Classification Report
                    st.subheader("📋 Classification Report")
                    report = classification_report(
                        y_true_enc, y_pred, target_names=class_names
                    )
                    st.code(report)

        except Exception as e:
            st.error(f"Error processing file: {e}")
    else:
        st.info("👆 Upload a CSV file to get started. Use test_data_sample.csv from the repository.")

# ===========================
# TAB 2: Metrics Comparison
# ===========================
with tab2:
    st.subheader("📊 Model Comparison — All 6 Models")
    st.dataframe(
        metrics_df.style.highlight_max(axis=0, color="#d4edda").format("{:.2f}"),
        use_container_width=True
    )
    st.caption("Green highlights = best value per metric. Accuracy/AUC/Precision/Recall/F1 in %. MCC is 0-1 scale.")

    # Bar chart
    st.subheader("📈 Visual Comparison")
    metric_choice = st.selectbox(
        "Select metric to visualize",
        ["Accuracy", "AUC", "Precision", "Recall", "F1 Score", "MCC"]
    )
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(metrics_df)))
    bars = ax2.bar(metrics_df.index, metrics_df[metric_choice], color=colors,
                   edgecolor="black", linewidth=0.7)
    for bar, val in zip(bars, metrics_df[metric_choice]):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + (0.002 if metric_choice == "MCC" else 0.2),
                 f"{val:.2f}", ha="center", fontsize=9, fontweight="bold")
    ax2.set_title(f"{metric_choice} — All Models", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Model")
    ax2.set_ylabel(metric_choice)
    ax2.tick_params(axis="x", rotation=20)
    ax2.grid(axis="y", alpha=0.3)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig2)

# ===========================
# TAB 3: About Dataset
# ===========================
with tab3:
    st.subheader("ℹ️ About the Dry Bean Dataset")
    st.markdown("""
    | Property | Value |
    |---|---|
    | **Source** | UCI Machine Learning Repository |
    | **URL** | https://archive.ics.uci.edu/dataset/602/dry+bean+dataset |
    | **Instances** | 13,611 |
    | **Features** | 16 geometric shape features |
    | **Target Classes** | 7 (BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SEKER, SIRA) |
    | **Problem Type** | Multi-class Classification |
    | **Missing Values** | None |

    ### Feature Descriptions
    All 16 features are continuous geometric measurements extracted from bean images:

    Area, Perimeter, MajorAxisLength, MinorAxisLength, AspectRation, Eccentricity,
    ConvexArea, EquivDiameter, Extent, Solidity, roundness, Compactness,
    ShapeFactor1, ShapeFactor2, ShapeFactor3, ShapeFactor4

    ### Reference
    Koklu, M. and Ozkan, I.A., (2020). *Multiclass Classification of Dry Beans Using
    Computer Vision and Machine Learning Techniques.*
    Computers and Electronics in Agriculture, 174, 105507.
    """)

    st.subheader("🏆 Best Performing Model")
    best_model = metrics_df["Accuracy"].idxmax()
    best_acc   = metrics_df["Accuracy"].max()
    st.success(f"**{best_model}** achieved the highest accuracy of **{best_acc:.2f}%**")
