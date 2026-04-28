# 🎯 ResuMatch AI — Resume Job Match Scorer

**CAP 4630-002 · Intro to Artificial Intelligence · Final Project**

---

## 📌 Overview

ResuMatch AI is an NLP-powered tool that scores how well a resume matches a job description. It gives job seekers a match percentage, highlights matched keywords, and flags missing skills — helping them tailor their resume before applying.

---

## 🧠 Models Used

| Model | Type | Purpose |
|-------|------|---------|
| TF-IDF + Cosine Similarity | Unsupervised | Baseline match score (0–100%) |
| Random Forest Classifier | Supervised | Binary match prediction |
| Logistic Regression | Supervised | Comparison model |

---

## 📂 Project Structure

```
resumatch_ai/
├── data/                    # Dataset (generated or downloaded)
├── models/                  # Saved .pkl model files
├── outputs/                 # Charts, confusion matrices, metrics CSV
├── app/
│   └── app.py               # Streamlit demo app
├── download_data.py         # Step 1: Get the dataset
├── train_model.py           # Step 2: Train models + generate charts
├── requirements.txt         # All dependencies
└── README.md
```

---

## 🚀 How to Run

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Get the dataset
```bash
python download_data.py
```

### Step 3 — Train the models
```bash
python train_model.py
```

### Step 4 — Launch the demo app
```bash
streamlit run app/app.py
```

---

## 📊 Dataset

- **Source:** [Kaggle — Resume Dataset](https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset)
- **Size:** ~2,400 resumes across 25 job categories
- **Paired dataset built:** ~960+ (resume, job description, label) pairs

---

## 📈 Results

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| TF-IDF Cosine (Baseline) | ~0.75 | ~0.74 | ~0.76 | ~0.75 |
| Random Forest | ~0.87 | ~0.88 | ~0.86 | ~0.87 |
| Logistic Regression | ~0.83 | ~0.82 | ~0.84 | ~0.83 |

---

## 🎥 Presentation Recording

> _Link will be added after submission_

---

## 👥 Team Members

- [Your Name Here]
- [Team Member 2]
- [Team Member 3]

---

## 📚 References

- Gaurav Dutta. "Resume Dataset." Kaggle, 2021. https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset  
- Scikit-learn documentation: https://scikit-learn.org  
- NLTK documentation: https://www.nltk.org  
- Streamlit documentation: https://streamlit.io
