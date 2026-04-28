"""
ResuMatch AI — Resume Job Match Scorer
CAP 4630-002 Final Project

This script:
1. Loads resume + job description dataset
2. Preprocesses text
3. Trains TF-IDF + Cosine Similarity model (Baseline)
4. Trains a Random Forest classifier (Match / No Match)
5. Evaluates both models
6. Saves all charts and results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import re
import os
import pickle
import warnings
warnings.filterwarnings("ignore")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
from sklearn.preprocessing import LabelEncoder

# ─── Config ───────────────────────────────────────────────────────────────────
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Built-in stopwords (no network needed)
STOP_WORDS = {
    'i','me','my','myself','we','our','ours','ourselves','you','your','yours',
    'yourself','yourselves','he','him','his','himself','she','her','hers','herself',
    'it','its','itself','they','them','their','theirs','themselves','what','which',
    'who','whom','this','that','these','those','am','is','are','was','were','be',
    'been','being','have','has','had','having','do','does','did','doing','a','an',
    'the','and','but','if','or','because','as','until','while','of','at','by','for',
    'with','about','against','between','into','through','during','before','after',
    'above','below','to','from','up','down','in','out','on','off','over','under',
    'again','further','then','once','here','there','when','where','why','how','all',
    'both','each','few','more','most','other','some','such','no','nor','not','only',
    'own','same','so','than','too','very','s','t','can','will','just','don','should',
    'now','d','ll','m','o','re','ve','y','ain','aren','couldn','didn','doesn','hadn',
    'hasn','haven','isn','ma','mightn','mustn','needn','shan','shouldn','wasn',
    'weren','won','wouldn','also','would','could','may','might','shall','us'
}

def simple_lemmatize(word):
    """Simple rule-based lemmatization."""
    if word.endswith('ing') and len(word) > 6:
        return word[:-3]
    if word.endswith('tion') and len(word) > 6:
        return word[:-4]
    if word.endswith('ed') and len(word) > 5:
        return word[:-2]
    if word.endswith('ly') and len(word) > 5:
        return word[:-2]
    if word.endswith('ies') and len(word) > 5:
        return word[:-3] + 'y'
    if word.endswith('s') and not word.endswith('ss') and len(word) > 4:
        return word[:-1]
    return word


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════

def load_data():
    """
    Load the Resume dataset from CSV.
    We create a paired dataset: (resume_text, job_description_text, label)
    label = 1 means GOOD MATCH, 0 means POOR MATCH
    """
    print("\n📂 Loading dataset...")

    df = pd.read_csv("data/resume_dataset.csv")
    print(f"   Raw shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Categories found: {df['Category'].nunique()}")
    print(df['Category'].value_counts().head(10))

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — TEXT PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def clean_text(text):
    """Clean and normalize text."""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)         # remove URLs
    text = re.sub(r'[^a-z\s]', ' ', text)              # remove non-alpha
    text = re.sub(r'\s+', ' ', text).strip()           # collapse whitespace
    tokens = text.split()
    tokens = [simple_lemmatize(w) for w in tokens if w not in STOP_WORDS and len(w) > 2]
    return ' '.join(tokens)


def preprocess(df):
    print("\n🧹 Preprocessing text...")
    df = df.copy()
    df['clean_resume'] = df['Resume'].apply(clean_text)
    df = df[df['clean_resume'].str.len() > 50].reset_index(drop=True)
    print(f"   Rows after cleaning: {len(df)}")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — BUILD JOB DESCRIPTION TEMPLATES
# ═══════════════════════════════════════════════════════════════════════════════

JOB_DESCRIPTIONS = {
    "Data Science": """
        data scientist machine learning python pandas numpy scikit learn tensorflow 
        deep learning neural network statistics data analysis sql visualization 
        model training feature engineering big data regression classification clustering
        natural language processing computer vision jupyter notebook git
    """,
    "Java Developer": """
        java developer spring boot microservices rest api maven gradle junit 
        object oriented programming design patterns sql database hibernate 
        git agile scrum backend development cloud aws docker kubernetes
    """,
    "Web Designing": """
        web designer html css javascript react angular bootstrap figma adobe xd 
        ui ux responsive design frontend development user interface photoshop 
        wireframe prototype animation sass node js
    """,
    "Python Developer": """
        python developer django flask rest api sql postgresql mongodb 
        git docker linux celery redis aws lambda microservices 
        unit testing pytest data structures algorithms object oriented
    """,
    "HR": """
        human resources recruiter talent acquisition onboarding payroll benefits 
        employee relations performance management training development 
        hris workday adp communication interpersonal organizational
    """,
    "Accountant": """
        accounting finance cpa gaap financial reporting tax reconciliation 
        excel quickbooks sap auditing budgeting forecasting accounts payable 
        accounts receivable general ledger balance sheet
    """,
    "Sales": """
        sales business development account management crm salesforce 
        lead generation negotiation client relations revenue target 
        cold calling pipeline management b2b b2c communication
    """,
    "Testing": """
        qa testing selenium automation junit testng api testing 
        bug tracking jira regression testing performance testing 
        load testing agile scrum defect management test cases
    """,
    "DevOps Engineer": """
        devops engineer aws azure gcp docker kubernetes jenkins ci cd 
        terraform ansible linux bash scripting monitoring prometheus 
        grafana git version control infrastructure automation
    """,
    "Network Security Engineer": """
        network security firewall vpn intrusion detection encryption 
        cisco router switch tcp ip ssl tls vulnerability assessment 
        penetration testing siem log analysis incident response
    """,
    "Mechanical Engineer": """
        mechanical engineer cad solidworks autocad ansys fea design 
        manufacturing thermodynamics fluid mechanics project management 
        prototyping material science lean six sigma
    """,
    "Arts": """
        graphic design illustration photography video editing adobe creative 
        indesign photoshop illustrator premiere after effects animation 
        typography branding content creation visual communication
    """
}


def build_paired_dataset(df):
    """
    For each resume, create:
    - 1 POSITIVE pair (resume + matching job description) → label 1
    - 1 NEGATIVE pair (resume + non-matching job description) → label 0
    """
    print("\n🔗 Building paired match dataset...")

    resumes = []
    job_descs = []
    labels = []
    categories = []

    available_cats = [c for c in df['Category'].unique() if c in JOB_DESCRIPTIONS]
    df_filtered = df[df['Category'].isin(available_cats)].reset_index(drop=True)

    for _, row in df_filtered.iterrows():
        cat = row['Category']
        resume_text = row['clean_resume']

        # Positive match
        pos_jd = clean_text(JOB_DESCRIPTIONS[cat])
        resumes.append(resume_text)
        job_descs.append(pos_jd)
        labels.append(1)
        categories.append(cat)

        # Negative match — pick a random different category
        other_cats = [c for c in available_cats if c != cat]
        neg_cat = np.random.choice(other_cats)
        neg_jd = clean_text(JOB_DESCRIPTIONS[neg_cat])
        resumes.append(resume_text)
        job_descs.append(neg_jd)
        labels.append(0)
        categories.append(cat)

    paired_df = pd.DataFrame({
        'resume': resumes,
        'job_description': job_descs,
        'label': labels,
        'category': categories
    })

    print(f"   Total pairs: {len(paired_df)}")
    print(f"   Match (1): {paired_df['label'].sum()} | No Match (0): {(paired_df['label']==0).sum()}")
    return paired_df


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — TFIDF + COSINE SIMILARITY (Baseline)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_cosine_scores(paired_df):
    print("\n📐 Computing TF-IDF Cosine Similarity scores...")

    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    all_texts = list(paired_df['resume']) + list(paired_df['job_description'])
    tfidf.fit(all_texts)

    resume_vecs = tfidf.transform(paired_df['resume'])
    jd_vecs = tfidf.transform(paired_df['job_description'])

    scores = []
    for i in range(len(paired_df)):
        score = cosine_similarity(resume_vecs[i], jd_vecs[i])[0][0]
        scores.append(round(score * 100, 2))

    paired_df = paired_df.copy()
    paired_df['cosine_score'] = scores

    # Use 0.5 threshold → classify
    threshold = paired_df[paired_df['label']==1]['cosine_score'].median() * 0.6
    paired_df['cosine_pred'] = (paired_df['cosine_score'] >= threshold).astype(int)

    acc = accuracy_score(paired_df['label'], paired_df['cosine_pred'])
    prec = precision_score(paired_df['label'], paired_df['cosine_pred'])
    rec = recall_score(paired_df['label'], paired_df['cosine_pred'])
    f1 = f1_score(paired_df['label'], paired_df['cosine_pred'])

    print(f"   Cosine Similarity Baseline:")
    print(f"   Accuracy:  {acc:.4f}")
    print(f"   Precision: {prec:.4f}")
    print(f"   Recall:    {rec:.4f}")
    print(f"   F1 Score:  {f1:.4f}")

    # Save vectorizer
    with open("models/tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(tfidf, f)

    return paired_df, tfidf, {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5 — RANDOM FOREST CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════════

def train_random_forest(paired_df, tfidf):
    print("\n🌲 Training Random Forest Classifier...")

    resume_vecs = tfidf.transform(paired_df['resume'])
    jd_vecs = tfidf.transform(paired_df['job_description'])

    # Feature: element-wise difference + product (captures interaction)
    from scipy.sparse import hstack
    X = hstack([
        abs(resume_vecs - jd_vecs),
        resume_vecs.multiply(jd_vecs)
    ])
    y = paired_df['label'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"   Random Forest Results:")
    print(f"   Accuracy:  {acc:.4f}")
    print(f"   Precision: {prec:.4f}")
    print(f"   Recall:    {rec:.4f}")
    print(f"   F1 Score:  {f1:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['No Match','Match'])}")

    with open("models/random_forest.pkl", "wb") as f:
        pickle.dump(rf, f)

    return rf, y_test, y_pred, cm, {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6 — LOGISTIC REGRESSION (comparison model)
# ═══════════════════════════════════════════════════════════════════════════════

def train_logistic_regression(paired_df, tfidf):
    print("\n📊 Training Logistic Regression...")

    resume_vecs = tfidf.transform(paired_df['resume'])
    jd_vecs = tfidf.transform(paired_df['job_description'])

    from scipy.sparse import hstack
    X = hstack([abs(resume_vecs - jd_vecs), resume_vecs.multiply(jd_vecs)])
    y = paired_df['label'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"   Accuracy: {acc:.4f} | F1: {f1:.4f}")

    with open("models/logistic_regression.pkl", "wb") as f:
        pickle.dump(lr, f)

    return lr, y_test, y_pred, {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 7 — VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_all(paired_df, rf_cm, cosine_metrics, rf_metrics, lr_metrics, df_raw):
    print("\n📊 Generating all visualizations...")

    plt.rcParams.update({'font.size': 12, 'figure.dpi': 150})
    colors = ['#2563EB', '#16A34A', '#DC2626', '#D97706', '#7C3AED']

    # ── 1. Category Distribution ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    cat_counts = df_raw['Category'].value_counts().head(12)
    bars = ax.bar(cat_counts.index, cat_counts.values, color=colors[0], edgecolor='white', linewidth=0.5)
    ax.set_title("Resume Distribution by Job Category", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Job Category")
    ax.set_ylabel("Number of Resumes")
    plt.xticks(rotation=35, ha='right')
    for bar, val in zip(bars, cat_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, str(val),
                ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/01_category_distribution.png")
    plt.close()
    print("   ✅ Chart 1: Category Distribution")

    # ── 2. Cosine Score Distribution ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    match = paired_df[paired_df['label']==1]['cosine_score']
    no_match = paired_df[paired_df['label']==0]['cosine_score']
    ax.hist(match, bins=40, alpha=0.7, color=colors[1], label='Match (Label=1)', edgecolor='white')
    ax.hist(no_match, bins=40, alpha=0.7, color=colors[2], label='No Match (Label=0)', edgecolor='white')
    ax.set_title("TF-IDF Cosine Similarity Score Distribution", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Cosine Similarity Score (%)")
    ax.set_ylabel("Frequency")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/02_cosine_score_distribution.png")
    plt.close()
    print("   ✅ Chart 2: Cosine Score Distribution")

    # ── 3. Confusion Matrix (Random Forest) ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Match', 'Match'],
                yticklabels=['No Match', 'Match'],
                linewidths=0.5, ax=ax)
    ax.set_title("Random Forest — Confusion Matrix", fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel("Actual Label")
    ax.set_xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/03_confusion_matrix_rf.png")
    plt.close()
    print("   ✅ Chart 3: Confusion Matrix")

    # ── 4. Model Comparison Bar Chart ─────────────────────────────────────────
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    cosine_vals = [cosine_metrics[k] for k in ['accuracy','precision','recall','f1']]
    rf_vals = [rf_metrics[k] for k in ['accuracy','precision','recall','f1']]
    lr_vals = [lr_metrics[k] for k in ['accuracy','precision','recall','f1']]

    x = np.arange(len(metrics_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 6))
    b1 = ax.bar(x - width, cosine_vals, width, label='TF-IDF Cosine (Baseline)', color=colors[3], edgecolor='white')
    b2 = ax.bar(x, rf_vals, width, label='Random Forest', color=colors[0], edgecolor='white')
    b3 = ax.bar(x + width, lr_vals, width, label='Logistic Regression', color=colors[4], edgecolor='white')

    ax.set_title("Model Performance Comparison", fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.set_ylim(0, 1.15)
    ax.legend()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f'{h:.2f}', xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/04_model_comparison.png")
    plt.close()
    print("   ✅ Chart 4: Model Comparison")

    # ── 5. Average Cosine Score by Category ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    avg_scores = paired_df[paired_df['label']==1].groupby('category')['cosine_score'].mean().sort_values(ascending=False)
    bars = ax.bar(avg_scores.index, avg_scores.values, color=colors[1], edgecolor='white')
    ax.set_title("Average Match Score by Job Category", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Job Category")
    ax.set_ylabel("Avg Cosine Similarity Score (%)")
    plt.xticks(rotation=35, ha='right')
    for bar, val in zip(bars, avg_scores.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/05_avg_score_by_category.png")
    plt.close()
    print("   ✅ Chart 5: Avg Score by Category")

    # ── 6. Label Balance Pie Chart ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 6))
    label_counts = paired_df['label'].value_counts()
    ax.pie(label_counts.values, labels=['Match', 'No Match'],
           autopct='%1.1f%%', colors=[colors[1], colors[2]],
           startangle=90, wedgeprops={'edgecolor': 'white', 'linewidth': 2})
    ax.set_title("Dataset Label Distribution", fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/06_label_distribution.png")
    plt.close()
    print("   ✅ Chart 6: Label Distribution")

    print(f"\n✅ All charts saved to ./{OUTPUT_DIR}/")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 8 — SAVE METRICS SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

def save_metrics_summary(cosine_metrics, rf_metrics, lr_metrics):
    summary = pd.DataFrame({
        'Model': ['TF-IDF Cosine Baseline', 'Random Forest', 'Logistic Regression'],
        'Accuracy': [cosine_metrics['accuracy'], rf_metrics['accuracy'], lr_metrics['accuracy']],
        'Precision': [cosine_metrics['precision'], rf_metrics['precision'], lr_metrics['precision']],
        'Recall': [cosine_metrics['recall'], rf_metrics['recall'], lr_metrics['recall']],
        'F1 Score': [cosine_metrics['f1'], rf_metrics['f1'], lr_metrics['f1']],
    })
    for col in ['Accuracy', 'Precision', 'Recall', 'F1 Score']:
        summary[col] = summary[col].apply(lambda x: f"{x:.4f}")

    summary.to_csv(f"{OUTPUT_DIR}/metrics_summary.csv", index=False)
    print("\n📋 Metrics Summary:")
    print(summary.to_string(index=False))
    print(f"\n✅ Saved to {OUTPUT_DIR}/metrics_summary.csv")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)

    # Run full pipeline
    df_raw = load_data()
    df_clean = preprocess(df_raw)
    paired_df = build_paired_dataset(df_clean)
    paired_df, tfidf, cosine_metrics = compute_cosine_scores(paired_df)
    rf, y_test_rf, y_pred_rf, rf_cm, rf_metrics = train_random_forest(paired_df, tfidf)
    lr, y_test_lr, y_pred_lr, lr_metrics = train_logistic_regression(paired_df, tfidf)
    plot_all(paired_df, rf_cm, cosine_metrics, rf_metrics, lr_metrics, df_raw)
    save_metrics_summary(cosine_metrics, rf_metrics, lr_metrics)

    print("\n🎉 Training complete! All models and charts saved.")
    print("   Next: run  python app/app.py  to launch the demo.")
