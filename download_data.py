"""
download_data.py — Downloads the Resume dataset from Kaggle
Run this FIRST before train_model.py

Requirements:
    pip install kaggle
    You need a Kaggle account and API key (instructions below)
"""

import os
import sys

def download_with_kaggle():
    """Try to download via kaggle API."""
    try:
        import kaggle
        print("📥 Downloading resume dataset from Kaggle...")
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            'gauravduttakiit/resume-dataset',
            path='data/',
            unzip=True
        )
        # Rename if needed
        for fname in os.listdir('data/'):
            if fname.lower().endswith('.csv') and 'resume' in fname.lower():
                src = os.path.join('data', fname)
                dst = os.path.join('data', 'resume_dataset.csv')
                if src != dst:
                    os.rename(src, dst)
                print(f"✅ Dataset saved as data/resume_dataset.csv")
                return True
        print("✅ Dataset downloaded to data/")
        return True
    except Exception as e:
        print(f"⚠️  Kaggle download failed: {e}")
        return False


def create_sample_dataset():
    """
    Create a realistic synthetic dataset for testing
    if Kaggle is not available.
    """
    import pandas as pd
    import random

    print("🔧 Creating synthetic resume dataset for demonstration...")

    categories = [
        "Data Science", "Java Developer", "Python Developer",
        "Web Designing", "HR", "Accountant", "Sales",
        "Testing", "DevOps Engineer", "Network Security Engineer",
        "Mechanical Engineer", "Arts"
    ]

    resume_templates = {
        "Data Science": [
            "Experienced data scientist skilled in Python pandas numpy scikit-learn TensorFlow machine learning deep learning. Worked on regression classification clustering models. Strong SQL and data visualization with Tableau. Used Jupyter notebooks for analysis. Experience with AWS and big data tools.",
            "Data analyst with expertise in statistical modeling predictive analytics Python R machine learning. Built neural network models for NLP computer vision tasks. Proficient in feature engineering model training evaluation. Published research on deep learning techniques.",
            "Machine learning engineer specializing in Python scikit-learn TensorFlow PyTorch. Developed and deployed production ML pipelines. Extensive experience with data preprocessing feature selection hyperparameter tuning. Strong background in statistics mathematics.",
        ],
        "Java Developer": [
            "Java software engineer with 4 years experience in Spring Boot microservices REST API development. Proficient in Maven Gradle JUnit design patterns. Worked with PostgreSQL MySQL databases. Experience with Docker Kubernetes CI/CD pipelines and Agile Scrum methodology.",
            "Backend Java developer specializing in enterprise applications Spring Framework Hibernate ORM. Built scalable REST APIs microservices architecture. Experience with cloud platforms AWS Azure. Strong object oriented programming skills.",
            "Senior Java developer with expertise in Spring Boot REST API microservices. Implemented design patterns like factory singleton observer. Extensive experience with SQL database optimization. Led teams using Agile Scrum methodology.",
        ],
        "Python Developer": [
            "Python developer experienced in Django Flask REST API development. Built web applications with PostgreSQL MongoDB backends. Proficient in Git Docker Linux environment. Experience with AWS Lambda serverless functions and celery task queues.",
            "Backend Python engineer specializing in Django REST framework. Developed scalable APIs deployed on AWS. Experience with Redis caching PostgreSQL optimization unit testing pytest. Strong Python programming skills and data structures knowledge.",
            "Python software engineer with expertise in Flask microservices Docker Kubernetes. Built ML model deployment pipelines. Proficient in SQL NoSQL databases. Experience with CI/CD Jenkins GitHub Actions.",
        ],
        "Web Designing": [
            "Frontend web designer skilled in HTML CSS JavaScript React Bootstrap. Created responsive UI/UX designs using Figma Adobe XD. Experience with animations CSS transitions. Built interactive web applications with modern JavaScript frameworks.",
            "UI UX designer and frontend developer proficient in React Angular HTML CSS JavaScript. Used Figma for wireframing prototyping. Experience with Sass Node.js REST API integration. Created accessible responsive web designs.",
            "Web designer with expertise in HTML CSS JavaScript Bootstrap React. Used Adobe Photoshop Illustrator for graphics. Experience with Figma for UI design user experience research. Built mobile-first responsive websites.",
        ],
        "HR": [
            "HR specialist with 5 years experience in talent acquisition recruiting onboarding payroll administration. Proficient in Workday ADP HRIS systems. Managed employee relations performance management training programs. PHR certified.",
            "Human resources manager experienced in full cycle recruiting employee onboarding benefits administration. Used HRIS platforms for payroll processing. Led performance review cycles training development initiatives. Strong interpersonal communication skills.",
            "HR generalist skilled in talent acquisition compensation benefits administration compliance. Experience with ADP Workday HR systems. Managed employee relations conflict resolution. Developed training programs for new hires.",
        ],
        "Accountant": [
            "CPA with expertise in GAAP financial reporting tax preparation reconciliation. Proficient in Excel QuickBooks SAP accounting software. Experience with auditing budgeting financial analysis. Strong attention to detail organizational skills.",
            "Senior accountant experienced in general ledger accounts payable accounts receivable financial statements. Used SAP Oracle for financial management. Prepared tax returns performed internal audits. CPA licensed with strong analytical skills.",
            "Accounting professional with expertise in financial reporting budgeting forecasting. Proficient in QuickBooks Excel pivot tables. Experience with accounts payable receivable reconciliation month end close. Bachelor degree in Accounting.",
        ],
        "Sales": [
            "Sales representative with 4 years B2B sales experience. Proficient in Salesforce CRM pipeline management. Consistently exceeded revenue targets through cold calling lead generation. Strong negotiation client relationship management skills.",
            "Account manager experienced in B2B B2C sales business development. Used CRM tools for pipeline tracking. Built client relationships through consultative selling approach. Exceeded quarterly sales targets by 30 percent.",
            "Sales professional skilled in lead generation prospecting cold calling. Managed accounts using Salesforce CRM. Experience in SaaS product sales enterprise negotiations. Strong communication presentation skills.",
        ],
        "Testing": [
            "QA engineer experienced in Selenium automation testing Java TestNG JUnit. Performed regression testing performance testing API testing. Used JIRA for defect tracking bug reporting. Experience with Agile Scrum development cycles.",
            "Software tester skilled in manual and automated testing using Selenium Python. Created test cases test plans for web and mobile applications. Experience with load testing JMeter API testing Postman. Strong attention to detail.",
            "QA automation engineer proficient in Selenium WebDriver TestNG Java. Developed automated test frameworks for regression testing. Used JIRA Confluence for project management. Experience with CI/CD Jenkins integration testing.",
        ],
        "DevOps Engineer": [
            "DevOps engineer with expertise in AWS Docker Kubernetes Jenkins CI/CD pipelines. Implemented infrastructure as code using Terraform Ansible. Experience with Linux bash scripting monitoring Prometheus Grafana. Strong automation skills.",
            "Cloud DevOps professional skilled in AWS GCP Docker Kubernetes. Built and maintained CI/CD pipelines using Jenkins GitLab. Experience with infrastructure automation monitoring alerting. Proficient in Linux scripting.",
            "Site reliability engineer experienced in Kubernetes Docker AWS Azure. Implemented monitoring using Prometheus Grafana ELK stack. Built deployment pipelines Terraform infrastructure automation. Strong Linux systems administration.",
        ],
        "Network Security Engineer": [
            "Network security engineer with expertise in firewall configuration VPN intrusion detection systems. Proficient in Cisco router switch configuration TCP IP networking. Experience with penetration testing vulnerability assessment security audits.",
            "Cybersecurity professional skilled in network security firewall VPN SSL TLS encryption. Performed vulnerability assessments penetration testing incident response. Experience with SIEM tools log analysis threat detection.",
            "Information security engineer experienced in network security firewall management IDS IPS. Conducted security audits risk assessments. Proficient in Cisco networking TCP IP protocols. Strong knowledge of encryption standards.",
        ],
        "Mechanical Engineer": [
            "Mechanical engineer with expertise in CAD SolidWorks AutoCAD ANSYS finite element analysis. Worked on product design manufacturing processes thermodynamics fluid mechanics. Experience with project management lean manufacturing six sigma.",
            "Design engineer skilled in SolidWorks AutoCAD CAE simulation. Developed mechanical components for aerospace automotive industries. Experience with material science prototyping testing validation. Lean six sigma green belt certified.",
            "Mechanical engineer experienced in product design CAD modeling FEA simulation. Used SolidWorks ANSYS for structural thermal analysis. Background in manufacturing processes quality control. Strong problem solving analytical skills.",
        ],
        "Arts": [
            "Graphic designer skilled in Adobe Photoshop Illustrator InDesign Premiere Pro. Created brand identities marketing materials digital content. Experience with typography layout design print media. Strong visual communication creativity.",
            "Creative designer proficient in Adobe Creative Suite illustration photography video editing. Developed visual content for social media digital marketing. Experience with UI design branding logo creation. Portfolio of diverse design projects.",
            "Multimedia designer experienced in graphic design video production photography. Proficient in Adobe After Effects Premiere Photoshop Illustrator. Created animations motion graphics for digital campaigns. Strong artistic creative skills.",
        ]
    }

    rows = []
    for cat, templates in resume_templates.items():
        for _ in range(40):
            base = random.choice(templates)
            # Add some variation
            extras = [
                "Bachelor degree in related field.", "5 years industry experience.",
                "Strong communication teamwork skills.", "Seeking challenging opportunities.",
                "Detail oriented problem solver.", "Certified professional.",
                "Proven track record of success.", "Results driven individual.",
            ]
            text = base + " " + " ".join(random.sample(extras, 2))
            rows.append({'Category': cat, 'Resume': text})

    df = pd.DataFrame(rows)
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/resume_dataset.csv', index=False)
    print(f"✅ Synthetic dataset created: data/resume_dataset.csv ({len(df)} rows, {df['Category'].nunique()} categories)")
    return True


if __name__ == "__main__":
    os.makedirs('data', exist_ok=True)

    # Try Kaggle first; fall back to synthetic
    success = download_with_kaggle()
    if not success:
        print("\n📌 Kaggle download unavailable. Using synthetic dataset instead.")
        print("   (This is fine for the project — results will still demonstrate the full pipeline)\n")
        create_sample_dataset()

    print("\n✅ Data ready! Now run: python train_model.py")
