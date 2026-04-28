"""
ResuMatch AI — Streamlit Demo App
CAP 4630-002 Final Project

Run with: streamlit run app/app.py
"""

import streamlit as st
import pickle
import re
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ResuMatch AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #2563EB, #7C3AED);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center;
        color: #6B7280;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .score-box {
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .score-high   { background: #DCFCE7; border: 2px solid #16A34A; }
    .score-medium { background: #FEF9C3; border: 2px solid #CA8A04; }
    .score-low    { background: #FEE2E2; border: 2px solid #DC2626; }
    .metric-card {
        background: #F8FAFC;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #E2E8F0;
    }
    .keyword-match   { background:#DCFCE7; color:#166534; padding:4px 10px; border-radius:20px; margin:3px; display:inline-block; font-size:0.85rem; }
    .keyword-missing { background:#FEE2E2; color:#991B1B; padding:4px 10px; border-radius:20px; margin:3px; display:inline-block; font-size:0.85rem; }
    .stTextArea textarea { font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

# ─── Load Models ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        with open(os.path.join(base, "models/tfidf_vectorizer.pkl"), "rb") as f:
            tfidf = pickle.load(f)
        with open(os.path.join(base, "models/random_forest.pkl"), "rb") as f:
            rf = pickle.load(f)
        return tfidf, rf
    except FileNotFoundError:
        return None, None

tfidf, rf = load_models()

# ─── Text Cleaning ────────────────────────────────────────────────────────────
STOP_WORDS = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in STOP_WORDS and len(w) > 2]
    return ' '.join(tokens)

def compute_score(resume_text, jd_text):
    from sklearn.metrics.pairwise import cosine_similarity
    resume_clean = clean_text(resume_text)
    jd_clean = clean_text(jd_text)
    r_vec = tfidf.transform([resume_clean])
    j_vec = tfidf.transform([jd_clean])
    score = cosine_similarity(r_vec, j_vec)[0][0] * 100
    return round(score, 1), resume_clean, jd_clean

def get_keyword_analysis(resume_text, jd_text, top_n=20):
    jd_words = set(clean_text(jd_text).split())
    resume_words = set(clean_text(resume_text).split())
    # Filter to meaningful words (longer than 3 chars)
    jd_keywords = {w for w in jd_words if len(w) > 3}
    matched = sorted(jd_keywords & resume_words)[:top_n]
    missing = sorted(jd_keywords - resume_words)[:top_n]
    return matched, missing

def get_score_label(score):
    if score >= 40: return "Strong Match 🟢", "score-high", "#16A34A"
    elif score >= 20: return "Partial Match 🟡", "score-medium", "#CA8A04"
    else: return "Weak Match 🔴", "score-low", "#DC2626"

def make_gauge(score):
    fig, ax = plt.subplots(figsize=(4, 2.2), subplot_kw=dict(polar=False))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    ax.axis('off')
    # Background bar
    ax.barh(0.5, 100, height=0.35, color='#E5E7EB', left=0, zorder=1)
    # Score bar
    color = '#16A34A' if score >= 40 else '#CA8A04' if score >= 20 else '#DC2626'
    ax.barh(0.5, score, height=0.35, color=color, left=0, zorder=2)
    ax.text(50, 0.12, f"{score:.1f}%", ha='center', va='center',
            fontsize=18, fontweight='bold', color='#1E293B')
    ax.text(50, 0.88, "Match Score", ha='center', va='center',
            fontsize=11, color='#6B7280')
    plt.tight_layout(pad=0.3)
    return fig

# ═══════════════════════════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="main-title">🎯 ResuMatch AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">CAP 4630-002 · Resume–Job Description Match Scorer</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/resume.png", width=70)
    st.markdown("### About ResuMatch AI")
    st.markdown("""
    This tool uses **NLP + Machine Learning** to score how well a resume matches a job description.

    **Models used:**
    - TF-IDF Vectorization
    - Cosine Similarity (Baseline)
    - Random Forest Classifier
    - Logistic Regression

    **CAP 4630-002**  
    Intro to Artificial Intelligence  
    Final Project
    """)
    st.divider()

    st.markdown("### 📋 Sample Inputs")
    if st.button("Load Data Science Example"):
        st.session_state['sample'] = 'ds'
    if st.button("Load Software Engineer Example"):
        st.session_state['sample'] = 'se'
    if st.button("Load HR Example"):
        st.session_state['sample'] = 'hr'

# Sample text
SAMPLES = {
    'ds': {
        'resume': """John Smith | Data Scientist
Skills: Python, pandas, numpy, scikit-learn, TensorFlow, machine learning, deep learning, 
data analysis, SQL, Tableau, Jupyter Notebook, statistics, regression, classification, 
clustering, NLP, computer vision, Git, AWS, feature engineering, model deployment.
Experience: 3 years at TechCorp building ML models for fraud detection. 
MS in Computer Science, University of Florida.""",
        'jd': """Senior Data Scientist — TechCorp
We are looking for a Data Scientist with expertise in Python, machine learning, and deep learning.
Requirements: Experience with pandas, scikit-learn, TensorFlow or PyTorch. Strong statistics background.
Must know SQL, data visualization (Tableau or Power BI), and have experience with cloud platforms (AWS/GCP).
Knowledge of NLP and computer vision a plus. MS or PhD preferred."""
    },
    'se': {
        'resume': """Jane Doe | Software Engineer
Skills: Java, Spring Boot, REST API, microservices, SQL, PostgreSQL, Git, Docker, Kubernetes,
Agile, Scrum, design patterns, unit testing, JUnit, Maven.
Experience: 2 years building backend services at StartupXYZ.""",
        'jd': """Backend Software Engineer — FinTech Inc
Requirements: Java or Kotlin, Spring Framework, REST API design, microservices architecture.
Must have experience with SQL databases (PostgreSQL or MySQL), Docker, Kubernetes, CI/CD pipelines.
Bonus: AWS experience, Kafka, Redis. 2+ years experience required. BS in Computer Science."""
    },
    'hr': {
        'resume': """Maria Garcia | HR Specialist
Skills: talent acquisition, onboarding, payroll, HRIS, Workday, ADP, performance management,
employee relations, training and development, communication, compliance.
5 years HR experience in healthcare sector. PHR Certified.""",
        'jd': """HR Manager — Healthcare Organization
We need an experienced HR professional for talent acquisition, employee onboarding, and benefits administration.
HRIS experience (Workday preferred), payroll management, performance reviews, and HR compliance required.
PHR or SPHR certification strongly preferred. 4+ years experience."""
    }
}

# Autofill samples
default_resume = ""
default_jd = ""
if 'sample' in st.session_state:
    s = st.session_state['sample']
    default_resume = SAMPLES[s]['resume']
    default_jd = SAMPLES[s]['jd']

# Input columns
col1, col2 = st.columns(2)
with col1:
    st.markdown("#### 📄 Paste Your Resume")
    resume_input = st.text_area(
        "Resume Text", value=default_resume,
        height=280, placeholder="Paste the full resume text here...",
        label_visibility="collapsed"
    )

with col2:
    st.markdown("#### 💼 Paste Job Description")
    jd_input = st.text_area(
        "Job Description", value=default_jd,
        height=280, placeholder="Paste the job description here...",
        label_visibility="collapsed"
    )

# Score button
st.markdown("")
_, btn_col, _ = st.columns([2, 1, 2])
with btn_col:
    analyze = st.button("🔍 Analyze Match", use_container_width=True, type="primary")

# ─── Results ──────────────────────────────────────────────────────────────────
if analyze:
    if not resume_input.strip() or not jd_input.strip():
        st.error("⚠️ Please paste both a resume and a job description.")
    elif tfidf is None:
        st.error("⚠️ Models not found. Please run `python train_model.py` first.")
    else:
        with st.spinner("Analyzing..."):
            score, resume_clean, jd_clean = compute_score(resume_input, jd_input)
            label, css_class, color = get_score_label(score)
            matched_kw, missing_kw = get_keyword_analysis(resume_input, jd_input)

        st.divider()
        st.markdown("## 📊 Analysis Results")

        res_col, gauge_col, stats_col = st.columns([1, 1.5, 1])

        with res_col:
            st.markdown(f"""
            <div class="score-box {css_class}">
                <div style="font-size:3.5rem; font-weight:900; color:{color};">{score:.1f}%</div>
                <div style="font-size:1.2rem; font-weight:700; color:{color}; margin-top:0.5rem;">{label}</div>
            </div>
            """, unsafe_allow_html=True)

        with gauge_col:
            fig = make_gauge(score)
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with stats_col:
            st.markdown("**Resume word count:**")
            st.markdown(f"`{len(resume_input.split())} words`")
            st.markdown("**JD word count:**")
            st.markdown(f"`{len(jd_input.split())} words`")
            st.markdown("**Keywords matched:**")
            st.markdown(f"`{len(matched_kw)} / {len(matched_kw)+len(missing_kw)}`")

        # Keyword Analysis
        st.divider()
        kw_col1, kw_col2 = st.columns(2)

        with kw_col1:
            st.markdown(f"#### ✅ Matched Keywords ({len(matched_kw)})")
            if matched_kw:
                kw_html = ' '.join([f'<span class="keyword-match">{w}</span>' for w in matched_kw])
                st.markdown(kw_html, unsafe_allow_html=True)
            else:
                st.info("No strong keyword matches found.")

        with kw_col2:
            st.markdown(f"#### ❌ Missing Keywords ({len(missing_kw)})")
            if missing_kw:
                kw_html = ' '.join([f'<span class="keyword-missing">{w}</span>' for w in missing_kw])
                st.markdown(kw_html, unsafe_allow_html=True)
            else:
                st.success("Great! Your resume covers the key terms.")

        # Recommendations
        st.divider()
        st.markdown("#### 💡 Recommendations")
        if score >= 40:
            st.success("🎉 **Strong match!** Your resume aligns well with this job description. Consider tailoring your summary to echo the job title directly.")
        elif score >= 20:
            st.warning(f"⚡ **Decent match.** Add these missing keywords to improve your score: **{', '.join(missing_kw[:8])}**")
        else:
            st.error(f"📝 **Weak match.** Your resume needs significant tailoring. Focus on adding: **{', '.join(missing_kw[:10])}**")
