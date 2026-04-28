"""
ResuMatch AI — Resume Job Match Scorer
CAP 4630-002 Final Project

Real Dataset: https://www.kaggle.com/datasets/saugataroyarghya/resume-dataset
9,544 resumes | 28 job categories | real matched_score column
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import re, os, pickle, warnings
warnings.filterwarnings("ignore")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

print("🚀 ResuMatch AI — Training Pipeline Starting...")
os.makedirs("outputs", exist_ok=True)
os.makedirs("models",  exist_ok=True)

STOP_WORDS = {
    'i','me','my','we','our','you','your','he','him','his','she','her','it',
    'its','they','them','their','what','which','who','this','that','these',
    'those','am','is','are','was','were','be','been','being','have','has',
    'had','do','does','did','a','an','the','and','but','if','or','as','at',
    'by','for','with','about','into','through','during','before','after','to',
    'from','up','in','out','on','off','then','here','there','when','where',
    'all','both','each','more','most','other','some','no','not','only','own',
    'so','than','too','very','can','will','just','should','now','also','would',
    'could','may','might','shall','us','ll','ve','re','don',
}

def clean_text(text):
    if pd.isna(text) or str(text).strip() in ('', 'nan', 'None', '[]'):
        return ""
    text = str(text).lower()
    text = re.sub(r"[\[\]'\"{}()]", ' ', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = [w for w in text.split() if w not in STOP_WORDS and len(w) > 2]
    return ' '.join(tokens)

# STEP 1 — LOAD
def load_and_build():
    print("\n📂 Loading real Kaggle dataset...")
    df = pd.read_csv("data/resume_dataset.csv")
    df.columns = df.columns.str.replace('\ufeff', '', regex=False).str.strip()
    print(f"   Shape: {df.shape} | Categories: {df['job_position_name'].nunique()}")

    def build_resume(row):
        return ' '.join([str(row.get('skills','')), str(row.get('career_objective','')),
                         str(row.get('responsibilities','')), str(row.get('positions',''))])
    def build_jd(row):
        return ' '.join([str(row.get('job_position_name','')), str(row.get('skills_required','')),
                         str(row.get('responsibilities.1','')), str(row.get('educationaL_requirements','')),
                         str(row.get('experiencere_requirement',''))])

    df['resume_text'] = df.apply(build_resume, axis=1)
    df['jd_text']     = df.apply(build_jd, axis=1)
    df['label']       = (df['matched_score'] >= 0.6).astype(int)
    print(f"   Match(1): {df['label'].sum()} | No Match(0): {(df['label']==0).sum()}")
    return df

# STEP 2 — PREPROCESS
def preprocess(df):
    print("\n🧹 Cleaning text...")
    df = df.copy()
    df['clean_resume'] = df['resume_text'].apply(clean_text)
    df['clean_jd']     = df['jd_text'].apply(clean_text)
    mask = (df['clean_resume'].str.len() > 20) & (df['clean_jd'].str.len() > 10)
    df = df[mask].reset_index(drop=True)
    print(f"   Rows after cleaning: {len(df)}")
    return df

# STEP 3 — COSINE SIMILARITY
def compute_cosine_scores(df):
    print("\n📐 Computing TF-IDF Cosine Similarity...")
    tfidf = TfidfVectorizer(max_features=8000, ngram_range=(1, 2))
    tfidf.fit(list(df['clean_resume']) + list(df['clean_jd']))
    rv = tfidf.transform(df['clean_resume'])
    jv = tfidf.transform(df['clean_jd'])
    scores = []
    for i in range(0, len(df), 500):
        scores.extend(cosine_similarity(rv[i:i+500], jv[i:i+500]).diagonal())
    df = df.copy()
    df['cosine_score'] = [round(s*100, 2) for s in scores]
    threshold = df[df['label']==1]['cosine_score'].median() * 0.55
    df['cosine_pred'] = (df['cosine_score'] >= threshold).astype(int)
    acc  = accuracy_score(df['label'],  df['cosine_pred'])
    prec = precision_score(df['label'], df['cosine_pred'], zero_division=0)
    rec  = recall_score(df['label'],    df['cosine_pred'], zero_division=0)
    f1   = f1_score(df['label'],        df['cosine_pred'], zero_division=0)
    print(f"   Accuracy:{acc:.4f} | Precision:{prec:.4f} | Recall:{rec:.4f} | F1:{f1:.4f}")
    with open("models/tfidf_vectorizer.pkl","wb") as f: pickle.dump(tfidf, f)
    return df, tfidf, {'accuracy':acc,'precision':prec,'recall':rec,'f1':f1}

# STEP 4 — RANDOM FOREST
def train_rf(df, tfidf):
    print("\n🌲 Training Random Forest...")
    from scipy.sparse import hstack
    rv = tfidf.transform(df['clean_resume'])
    jv = tfidf.transform(df['clean_jd'])
    X  = hstack([abs(rv-jv), rv.multiply(jv)])
    y  = df['label'].values
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    print(f"   Train:{Xtr.shape[0]} | Test:{Xte.shape[0]}")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(Xtr, ytr); yp = rf.predict(Xte)
    acc=accuracy_score(yte,yp); prec=precision_score(yte,yp,zero_division=0)
    rec=recall_score(yte,yp,zero_division=0); f1=f1_score(yte,yp,zero_division=0)
    cm=confusion_matrix(yte,yp)
    print(f"   Accuracy:{acc:.4f} | Precision:{prec:.4f} | Recall:{rec:.4f} | F1:{f1:.4f}")
    print(classification_report(yte,yp,target_names=['No Match','Match']))
    with open("models/random_forest.pkl","wb") as f: pickle.dump(rf, f)
    return rf, yte, yp, cm, {'accuracy':acc,'precision':prec,'recall':rec,'f1':f1}

# STEP 5 — LOGISTIC REGRESSION
def train_lr(df, tfidf):
    print("\n📊 Training Logistic Regression...")
    from scipy.sparse import hstack
    rv = tfidf.transform(df['clean_resume'])
    jv = tfidf.transform(df['clean_jd'])
    X  = hstack([abs(rv-jv), rv.multiply(jv)])
    y  = df['label'].values
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(Xtr, ytr); yp = lr.predict(Xte)
    acc=accuracy_score(yte,yp); prec=precision_score(yte,yp,zero_division=0)
    rec=recall_score(yte,yp,zero_division=0); f1=f1_score(yte,yp,zero_division=0)
    print(f"   Accuracy:{acc:.4f} | F1:{f1:.4f}")
    with open("models/logistic_regression.pkl","wb") as f: pickle.dump(lr, f)
    return lr, yte, yp, {'accuracy':acc,'precision':prec,'recall':rec,'f1':f1}

# STEP 6 — CHARTS
def plot_all(df, cm, cm_, rf_, lr_):
    print("\n📊 Generating charts...")
    BLUE,GREEN,RED,AMBER,PURPLE='#2563EB','#16A34A','#DC2626','#D97706','#7C3AED'
    plt.rcParams.update({'font.size':11,'figure.dpi':150})

    # 1 Category distribution
    fig,ax=plt.subplots(figsize=(14,5))
    cats=df['job_position_name'].value_counts().head(15)
    bars=ax.bar(cats.index,cats.values,color=BLUE,edgecolor='white')
    ax.set_title("Resume Count by Job Category (Top 15)",fontsize=14,fontweight='bold',pad=12)
    ax.set_xlabel("Job Category"); ax.set_ylabel("Count")
    plt.xticks(rotation=38,ha='right',fontsize=8)
    for b,v in zip(bars,cats.values): ax.text(b.get_x()+b.get_width()/2,b.get_height()+1,str(v),ha='center',fontsize=7)
    plt.tight_layout(); plt.savefig("outputs/01_category_distribution.png"); plt.close()
    print("   ✅ Chart 1")

    # 2 Cosine score distribution
    fig,ax=plt.subplots(figsize=(10,5))
    ax.hist(df[df['label']==1]['cosine_score'],bins=40,alpha=0.7,color=GREEN,label='Match (label=1)',edgecolor='white')
    ax.hist(df[df['label']==0]['cosine_score'],bins=40,alpha=0.7,color=RED,label='No Match (label=0)',edgecolor='white')
    ax.set_title("TF-IDF Cosine Score Distribution by Label",fontsize=14,fontweight='bold',pad=12)
    ax.set_xlabel("Cosine Similarity Score (%)"); ax.set_ylabel("Frequency"); ax.legend()
    plt.tight_layout(); plt.savefig("outputs/02_cosine_score_distribution.png"); plt.close()
    print("   ✅ Chart 2")

    # 3 Confusion matrix
    fig,ax=plt.subplots(figsize=(7,6))
    sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=['No Match','Match'],
                yticklabels=['No Match','Match'],linewidths=0.5,ax=ax,annot_kws={"size":16})
    ax.set_title("Random Forest — Confusion Matrix",fontsize=14,fontweight='bold',pad=12)
    ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")
    plt.tight_layout(); plt.savefig("outputs/03_confusion_matrix_rf.png"); plt.close()
    print("   ✅ Chart 3")

    # 4 Model comparison
    metrics=['Accuracy','Precision','Recall','F1 Score']
    cv=[cm_[k] for k in ['accuracy','precision','recall','f1']]
    rv=[rf_[k] for k in ['accuracy','precision','recall','f1']]
    lv=[lr_[k] for k in ['accuracy','precision','recall','f1']]
    x,w=np.arange(4),0.25
    fig,ax=plt.subplots(figsize=(11,6))
    b1=ax.bar(x-w,cv,w,label='TF-IDF Cosine (Baseline)',color=AMBER,edgecolor='white')
    b2=ax.bar(x,rv,w,label='Random Forest',color=BLUE,edgecolor='white')
    b3=ax.bar(x+w,lv,w,label='Logistic Regression',color=PURPLE,edgecolor='white')
    ax.set_title("Model Performance Comparison",fontsize=14,fontweight='bold',pad=12)
    ax.set_ylabel("Score"); ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.set_ylim(0,1.15); ax.legend()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_:f'{v:.0%}'))
    for bars in [b1,b2,b3]:
        for b in bars:
            ax.annotate(f'{b.get_height():.2f}',xy=(b.get_x()+b.get_width()/2,b.get_height()),
                        xytext=(0,3),textcoords='offset points',ha='center',fontsize=8)
    plt.tight_layout(); plt.savefig("outputs/04_model_comparison.png"); plt.close()
    print("   ✅ Chart 4")

    # 5 Avg score by category
    fig,ax=plt.subplots(figsize=(13,5))
    avg=df.groupby('job_position_name')['cosine_score'].mean().sort_values(ascending=False).head(12)
    bars=ax.bar(avg.index,avg.values,color=GREEN,edgecolor='white')
    ax.set_title("Average Match Score by Job Category (Top 12)",fontsize=14,fontweight='bold',pad=12)
    ax.set_xlabel("Job Category"); ax.set_ylabel("Avg Cosine Score (%)")
    plt.xticks(rotation=35,ha='right',fontsize=8)
    for b,v in zip(bars,avg.values): ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.1,f'{v:.1f}%',ha='center',fontsize=8)
    plt.tight_layout(); plt.savefig("outputs/05_avg_score_by_category.png"); plt.close()
    print("   ✅ Chart 5")

    # 6 Label pie
    fig,ax=plt.subplots(figsize=(6,6))
    counts=df['label'].value_counts()
    ax.pie(counts.values,labels=['Match','No Match'],autopct='%1.1f%%',
           colors=[GREEN,RED],startangle=90,wedgeprops={'edgecolor':'white','linewidth':2})
    ax.set_title("Dataset Label Distribution",fontsize=14,fontweight='bold',pad=12)
    plt.tight_layout(); plt.savefig("outputs/06_label_distribution.png"); plt.close()
    print("   ✅ Chart 6")

    # 7 Original matched_score histogram
    fig,ax=plt.subplots(figsize=(10,5))
    ax.hist(df['matched_score'],bins=50,color=PURPLE,edgecolor='white',alpha=0.85)
    ax.axvline(0.6,color=RED,linestyle='--',linewidth=2,label='Match threshold (0.6)')
    ax.set_title("Original Dataset Matched Score Distribution",fontsize=14,fontweight='bold',pad=12)
    ax.set_xlabel("Matched Score (provided in dataset)"); ax.set_ylabel("Frequency"); ax.legend()
    plt.tight_layout(); plt.savefig("outputs/07_original_score_distribution.png"); plt.close()
    print("   ✅ Chart 7")
    print("\n✅ All charts saved to ./outputs/")

def save_metrics(cm_, rf_, lr_):
    summary=pd.DataFrame({
        'Model':['TF-IDF Cosine Baseline','Random Forest','Logistic Regression'],
        'Accuracy':[cm_['accuracy'],rf_['accuracy'],lr_['accuracy']],
        'Precision':[cm_['precision'],rf_['precision'],lr_['precision']],
        'Recall':[cm_['recall'],rf_['recall'],lr_['recall']],
        'F1 Score':[cm_['f1'],rf_['f1'],lr_['f1']],
    })
    for col in ['Accuracy','Precision','Recall','F1 Score']:
        summary[col]=summary[col].apply(lambda x:f"{x:.4f}")
    summary.to_csv("outputs/metrics_summary.csv",index=False)
    print("\n📋 Metrics Summary:"); print(summary.to_string(index=False))

if __name__ == "__main__":
    df                           = load_and_build()
    df                           = preprocess(df)
    df, tfidf, cm_               = compute_cosine_scores(df)
    rf, yt_rf, yp_rf, rf_cm, rf_ = train_rf(df, tfidf)
    lr, yt_lr, yp_lr, lr_        = train_lr(df, tfidf)
    plot_all(df, rf_cm, cm_, rf_, lr_)
    save_metrics(cm_, rf_, lr_)
    print("\n🎉 Done! Run: streamlit run app/app.py")
