# healthhive_full.py
"""
HealthHive - Comprehensive Health Monitoring, ML/DL Risk Prediction,
and Patient Report Generator (visual + textual).
- CPU-friendly
- Non-prescriptive medicine-category suggestions only
- Saves artifacts/visuals to ./output/
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # force CPU
import re
import json
import uuid
from datetime import datetime, timedelta
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from collections import Counter

# ML/DL
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# NLP helpers
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure output folder
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------
# NLTK Setup (quiet)
# -----------------------
try:
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
except Exception:
    pass

_LEMM = WordNetLemmatizer()
_STOP = set(stopwords.words("english")) if "stopwords" in nltk.corpus.__dict__ else {"the","a","and","in","on","of","to"}

# -----------------------
# Synthetic dataset generator (realistic health monitoring)
# -----------------------
def generate_health_dataset(num_records_per_patient=50, patient_ids=None, start_dt=None):
    """
    Create synthetic health monitoring records.
    Returns a DataFrame with columns:
    date,time,heartRate,bloodPressureSystolic,bloodPressureDiastolic,temperature,
    oxygenSaturation,sleepHours,stepsCount,feelingTired,stressLevel,mood,
    painLevel,energyLevel,symptoms,notes,entry_id,patient_id,created_at,risk_label
    """
    if patient_ids is None:
        patient_ids = [10]  # default single patient
    if start_dt is None:
        start_dt = datetime(2025, 9, 1, 8, 0)

    moods = ["neutral","happy","sad","frustrated","anxious"]
    symptoms_list = [ [], ["Headache"], ["Fatigue"], ["Cough","Fever"], ["Chest Pain"], ["Dizziness"], ["Shortness of breath"] ]
    risk_levels = ["low","medium","high"]

    records = []
    for pid in patient_ids:
        base = start_dt + timedelta(days=random.randint(0,3))
        for i in range(num_records_per_patient):
            t = base + timedelta(minutes=30*i)
            # realistic ranges
            hr = random.randint(55,110)
            sys = random.randint(100,160)
            dia = random.randint(60,100)
            temp = round(random.uniform(96.8,101.5),1)
            spo2 = random.randint(88,100)
            sleep = round(random.uniform(4,9),1)
            steps = random.randint(0,12000)
            tired = random.randint(1,10)
            stress = random.randint(1,10)
            mood = random.choice(moods)
            pain = random.randint(0,10)
            energy = random.randint(1,10)
            symptoms = random.choice(symptoms_list)
            notes = "" if random.random() > 0.7 else random.choice(["Not feeling good","Mild headache","Short breath on exertion","Felt dizzy"])
            # simple risk heuristic to create labels for training:
            risk_score = 0
            if temp >= 100.4: risk_score += 2
            if hr > 100: risk_score += 1
            if sys >= 140 or dia >= 90: risk_score += 1
            if spo2 < 94: risk_score += 2
            if "Chest Pain" in symptoms or "Shortness of breath" in symptoms: risk_score += 3
            # map to label
            if risk_score >= 4:
                risk = "high"
            elif risk_score >= 2:
                risk = "medium"
            else:
                risk = "low"

            rec = {
                "date": t.strftime("%Y-%m-%d"),
                "time": t.strftime("%H:%M"),
                "heartRate": hr,
                "bloodPressureSystolic": sys,
                "bloodPressureDiastolic": dia,
                "temperature": temp,
                "oxygenSaturation": spo2,
                "sleepHours": sleep,
                "stepsCount": steps,
                "feelingTired": tired,
                "stressLevel": stress,
                "mood": mood,
                "painLevel": pain,
                "energyLevel": energy,
                "symptoms": symptoms,
                "notes": notes,
                "entry_id": str(uuid.uuid4()),
                "patient_id": pid,
                "created_at": t.isoformat(),
                "risk_label": risk
            }
            records.append(rec)
    df = pd.DataFrame(records)
    return df

# -----------------------
# Simple NLP cleaning for symptoms/notes
# -----------------------
def clean_text_simple(s):
    if pd.isna(s): return ""
    s = str(s).lower()
    s = re.sub(r"http\S+|www\S+|https\S+", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    tokens = s.split()
    tokens = [t for t in tokens if t not in _STOP and len(t)>1]
    tokens = [ _LEMM.lemmatize(t) for t in tokens ]
    return " ".join(tokens)

# -----------------------
# Model training pipeline (ML + DL)
# -----------------------
class HealthModelPipeline:
    def __init__(self, df):
        self.df = df.copy()
        # convert list symptoms to string
        self.df["symptoms_text"] = self.df["symptoms"].apply(lambda x: " ".join(x) if isinstance(x,list) else str(x))
        self.df["notes_text"] = self.df["notes"].fillna("").astype(str)
        self.df["text_all"] = (self.df["symptoms_text"] + " " + self.df["notes_text"]).apply(clean_text_simple)
        self.label_map = {"low":0,"medium":1,"high":2}
        self.df["y"] = self.df["risk_label"].map(self.label_map).astype(int)

        # features for ML: vitals + text TF-IDF
        self.ml_vectorizer = TfidfVectorizer(max_features=1000)
        # tokenizers for DL
        self.dl_tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")

    def prepare_features(self, test_size=0.2):
        X_num = self.df[["heartRate","bloodPressureSystolic","bloodPressureDiastolic","temperature","oxygenSaturation","sleepHours","stepsCount","feelingTired","stressLevel","painLevel","energyLevel"]].fillna(0)
        X_text = self.df["text_all"]
        # vectorize text
        X_tfidf = self.ml_vectorizer.fit_transform(X_text)
        # combine numeric and text (dense)
        X_dense = np.hstack([X_num.values, X_tfidf.toarray()])
        X_train, X_test, y_train, y_test = train_test_split(X_dense, self.df["y"].values, test_size=test_size, random_state=42, stratify=self.df["y"].values)
        self.X_train_ml, self.X_test_ml, self.y_train_ml, self.y_test_ml = X_train, X_test, y_train, y_test

        # Prepare DL features (use only text sequences for DL)
        texts = X_text.tolist()
        self.dl_tokenizer.fit_on_texts(texts)
        seqs = self.dl_tokenizer.texts_to_sequences(texts)
        maxlen = 40
        sequences = pad_sequences(seqs, maxlen=maxlen, padding="post", truncating="post")
        Xtr, Xte, ytr, yte = train_test_split(sequences, self.df["y"].values, test_size=test_size, random_state=42, stratify=self.df["y"].values)
        self.X_train_dl, self.X_test_dl, self.y_train_dl, self.y_test_dl = Xtr, Xte, ytr, yte
        self.dl_maxlen = maxlen

    def train_ml_models(self):
        # Random Forest
        self.rf = RandomForestClassifier(n_estimators=200, random_state=42)
        self.rf.fit(self.X_train_ml, self.y_train_ml)
        # Logistic Regression (multi-class)
        self.lr = LogisticRegression(max_iter=1000, multi_class="ovr")
        self.lr.fit(self.X_train_ml, self.y_train_ml)

        # Evaluate quick
        preds = self.rf.predict(self.X_test_ml)
        print("RF accuracy:", accuracy_score(self.y_test_ml, preds))
        print(classification_report(self.y_test_ml, preds, digits=3))

        preds_lr = self.lr.predict(self.X_test_ml)
        print("LR accuracy:", accuracy_score(self.y_test_ml, preds_lr))
        print(classification_report(self.y_test_ml, preds_lr, digits=3))

        # Save vectorizer + models
        joblib.dump(self.ml_vectorizer, os.path.join(OUTPUT_DIR, "tfidf_vectorizer.joblib"))
        joblib.dump(self.rf, os.path.join(OUTPUT_DIR, "rf_model.joblib"))
        joblib.dump(self.lr, os.path.join(OUTPUT_DIR, "lr_model.joblib"))

    def build_and_train_dl(self, epochs=8, batch=32):
        vocab = min(5000, len(self.dl_tokenizer.word_index)+1)
        model = Sequential([
            Embedding(input_dim=vocab, output_dim=64, input_length=self.dl_maxlen),
            Bidirectional(LSTM(64, return_sequences=False)),
            Dropout(0.4),
            Dense(32, activation="relu"),
            Dropout(0.3),
            Dense(3, activation="softmax")  # 3 classes: low/medium/high
        ])
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        es = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
        rl = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, min_lr=1e-5)
        model.fit(self.X_train_dl, self.y_train_dl, validation_data=(self.X_test_dl, self.y_test_dl), epochs=epochs, batch_size=batch, callbacks=[es,rl], verbose=1)
        # evaluate
        dl_preds = np.argmax(model.predict(self.X_test_dl), axis=1)
        print("DL accuracy:", accuracy_score(self.y_test_dl, dl_preds))
        print(classification_report(self.y_test_dl, dl_preds, digits=3))
        # save
        model.save(os.path.join(OUTPUT_DIR, "dl_bilstm.keras"))
        joblib.dump(self.dl_tokenizer, os.path.join(OUTPUT_DIR, "dl_tokenizer.joblib"))
        self.dl_model = model

# -----------------------
# Suggest medicine categories (SAFE, non-prescriptive), with citations
# -----------------------
# This mapping below is intentionally high-level: categories, not specific doses or prescriptions.
# Sources: Mayo Clinic, CDC, NHS, MedlinePlus (see assistant message citations)
def suggest_care_and_med_categories(latest):
    """
    latest: dict-like record with vitals & symptoms
    returns: suggestions list and urgent_flag boolean
    """
    suggestions = []
    urgent = False

    temp = latest.get("temperature", None)
    spo2 = latest.get("oxygenSaturation", None)
    sys = latest.get("bloodPressureSystolic", None)
    dia = latest.get("bloodPressureDiastolic", None)
    hr = latest.get("heartRate", None)
    symptoms = [s.lower() for s in (latest.get("symptoms") or [])]

    # Fever -> antipyretic (paracetamol/ibuprofen categories). See Mayo Clinic / MedlinePlus.
    if temp is not None and temp >= 100.4:
        suggestions.append("Antipyretic (fever reducer) — e.g., acetaminophen/paracetamol or ibuprofen (OTC category).")
    # Low oxygen -> urgent (seek immediate care)
    if spo2 is not None and spo2 < 94:
        suggestions.append("Low oxygen saturation — seek medical attention immediately (oxygen evaluation).")
        urgent = True
    # High BP or chest pain / SOB
    if sys is not None and sys >= 140 or dia is not None and dia >= 90:
        suggestions.append("Elevated blood pressure — consult clinician for antihypertensive review; urgent if symptoms present.")
    if "chest pain" in symptoms or "shortness of breath" in symptoms:
        suggestions.append("Chest pain / shortness of breath — **seek emergency care immediately**.")
        urgent = True
    # Cough/runny nose -> cough suppressant / expectorant / saline/humidifier
    if "cough" in symptoms:
        suggestions.append("Cough care: hydration, humidifier/saline; if needed, OTC cough suppressants (dextromethorphan) or expectorants (guaifenesin) — discuss with pharmacist.")
    # Allergy/runny nose -> antihistamine / decongestant categories
    if "sneezing" in symptoms or "runny" in symptoms or "allergy" in symptoms:
        suggestions.append("Antihistamine / decongestant categories for nasal symptoms (consult pharmacist).")
    # Headache / pain -> analgesic category
    if "headache" in symptoms or latest.get("painLevel",0) >= 6:
        suggestions.append("Analgesic (pain reliever) category — e.g., acetaminophen/paracetamol, NSAIDs (ibuprofen) — consult clinician if chronic.")
    # Fatigue / low energy -> rest, hydration, check for anemia/thyroid with clinician
    if "fatigue" in symptoms or latest.get("energyLevel",10) <= 3:
        suggestions.append("Fatigue: rest, hydration; if persistent, seek clinician (investigate anemia, thyroid, sleep).")
    # General note: OTC medicines can relieve symptoms but do not cure underlying infections; consult pharmacist/doctor.
    suggestions.append("Non-pharmacologic care: rest, hydration, humidifier, honey for cough (age >1), saline nasal rinses.")
    # Safety note
    suggestions.append("Important: these are categories only. Do NOT take new prescription medications without clinician recommendation. For children, pregnant people, or complex medical histories consult a clinician/pharmacist first.")
    return suggestions, urgent

# -----------------------
# Reporting & Visualization
# -----------------------
def generate_patient_report(df, patient_id, pipeline=None):
    """
    df: DataFrame of records
    patient_id: integer
    pipeline: HealthModelPipeline instance (optional) to run model predictions
    Produces:
      - PNG report saved to output/patient_{id}_report.png
      - Returns summary dict
    """
    patient_df = df[df["patient_id"]==patient_id].sort_values(["created_at"])
    if patient_df.empty:
        raise ValueError("No data for patient_id="+str(patient_id))
    latest = patient_df.iloc[-1].to_dict()

    # If pipeline provided, run ML and DL predictions on latest record
    ml_pred_label = None
    ml_pred_proba = None
    dl_pred_label = None
    dl_pred_proba = None
    if pipeline is not None:
        # Prepare ML vector for latest
        text = latest.get("symptoms",[])
        text_s = " ".join(text) if isinstance(text,list) else str(text)
        text_s = clean_text_simple(text_s + " " + str(latest.get("notes","")))
        vec = pipeline.ml_vectorizer.transform([text_s])
        # numeric features in same order as pipeline
        num = np.array([[ latest.get("heartRate",0),
                          latest.get("bloodPressureSystolic",0),
                          latest.get("bloodPressureDiastolic",0),
                          latest.get("temperature",0),
                          latest.get("oxygenSaturation",0),
                          latest.get("sleepHours",0),
                          latest.get("stepsCount",0),
                          latest.get("feelingTired",0),
                          latest.get("stressLevel",0),
                          latest.get("painLevel",0),
                          latest.get("energyLevel",0) ]])
        Xml = np.hstack([num, vec.toarray()])
        # RF
        if hasattr(pipeline, "rf"):
            p = pipeline.rf.predict_proba(Xml)[0] if hasattr(pipeline.rf, "predict_proba") else None
            if p is not None:
                ml_pred_proba = p.tolist()
                ml_pred_label = ["low","medium","high"][int(np.argmax(p))]
        # DL
        try:
            tok = pipeline.dl_tokenizer
            seq = pad_sequences(tok.texts_to_sequences([text_s]), maxlen=pipeline.dl_maxlen)
            dl_model = pipeline.dl_model
            probs = dl_model.predict(seq)[0]
            dl_pred_proba = probs.tolist()
            dl_pred_label = ["low","medium","high"][int(np.argmax(probs))]
        except Exception:
            pass

    # Generate suggestions (non-prescriptive)
    suggestions, urgent = suggest_care_and_med_categories(latest)

    # Create visual figure: vitals over time + latest snapshot
    fig, axes = plt.subplots(3,1, figsize=(8,12))
    sns.lineplot(data=patient_df, x="created_at", y="heartRate", ax=axes[0], marker="o")
    axes[0].set_title("Heart Rate over time")
    sns.lineplot(data=patient_df, x="created_at", y="temperature", ax=axes[1], marker="o", color="orange")
    axes[1].set_title("Temperature over time (°F)")
    sns.lineplot(data=patient_df, x="created_at", y="oxygenSaturation", ax=axes[2], marker="o", color="green")
    axes[2].set_title("Oxygen saturation over time (%)")
    plt.tight_layout()
    png_path = os.path.join(OUTPUT_DIR, f"patient_{patient_id}_vitals.png")
    plt.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Create text report JSON and a small summary PNG with recommendations
    report = {
        "patient_id": patient_id,
        "latest_snapshot": latest,
        "ml_prediction": {"label": ml_pred_label, "proba": ml_pred_proba},
        "dl_prediction": {"label": dl_pred_label, "proba": dl_pred_proba},
        "suggestions": suggestions,
        "urgent": urgent,
        "visualization": png_path,
        "generated_at": datetime.now().isoformat()
    }

    # Create summary PNG with textual recommendations
    fig2, ax2 = plt.subplots(figsize=(8,6))
    ax2.axis("off")
    text_lines = [
        f"HealthHive Patient Report — ID: {patient_id}",
        f"Generated: {report['generated_at']}",
        "",
        f"Latest vitals (temp/HR/SpO2): {latest.get('temperature')}°F / {latest.get('heartRate')} bpm / {latest.get('oxygenSaturation')}%",
        f"Latest symptoms: {', '.join(latest.get('symptoms') or []) or 'None'}",
        "",
        f"ML prediction: {ml_pred_label} {('('+str(ml_pred_proba)+')') if ml_pred_proba else ''}",
        f"DL prediction: {dl_pred_label} {('('+str(dl_pred_proba)+')') if dl_pred_proba else ''}",
        "",
        "Suggestions:",
    ]
    text_lines += [f"- {s}" for s in suggestions[:6]]
    text_lines += ["", "⚠️ This is NOT medical advice. Consult a clinician."]
    ax2.text(0,1, "\n".join(text_lines), va="top", fontsize=10, family="monospace")
    summary_png = os.path.join(OUTPUT_DIR, f"patient_{patient_id}_report_summary.png")
    plt.savefig(summary_png, bbox_inches="tight", dpi=200)
    plt.close(fig2)

    # Save JSON report
    json_path = os.path.join(OUTPUT_DIR, f"patient_{patient_id}_report.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    return report

# -----------------------
# Example run (main)
# -----------------------
def main():
    print("HealthHive demo: generate dataset, train, and produce patient report")
    # generate dataset for patients 10 and 11
    df = generate_health_dataset(num_records_per_patient=30, patient_ids=[10,11], start_dt=datetime(2025,9,1,8,0))
    df.to_csv(os.path.join(OUTPUT_DIR,"health_dataset.csv"), index=False)
    print("Dataset saved to output/health_dataset.csv")

    # pipeline train
    pipeline = HealthModelPipeline(df)
    pipeline.prepare_features(test_size=0.2)
    pipeline.train_ml_models()
    pipeline.build_and_train_dl(epochs=6, batch=32)

    # Generate report for patient 10
    report = generate_patient_report(df, patient_id=10, pipeline=pipeline)
    print("Report generated:", os.path.join(OUTPUT_DIR, f"patient_{10}_report.json"))
    print("Summary image:", report["visualization"])
    if report["urgent"]:
        print("\n*** URGENT: Patient flagged with urgent condition. Recommend immediate clinical attention! ***")

if __name__ == "__main__":
    main()
