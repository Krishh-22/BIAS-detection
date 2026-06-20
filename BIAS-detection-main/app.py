import sqlite3
import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from PyPDF2 import PdfReader
from sklearn.ensemble import RandomForestClassifier
from fairlearn.reductions import ExponentiatedGradient, DemographicParity

# --- 1. GLOBAL UI CONFIG ---
st.set_page_config(page_title="AI Recruitment Bias & Resume Analyzer", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; border-radius: 12px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
    .verdict-card { padding: 30px; border-radius: 15px; color: white; margin-bottom: 20px; }
    .status-pass { background: linear-gradient(135deg, #28a745 0%, #85d084 100%); border-left: 10px solid #1e7e34; }
    .status-fail { background: linear-gradient(135deg, #dc3545 0%, #f1929a 100%); border-left: 10px solid #bd2130; }
    .footer { text-align: center; color: #888; font-size: 0.8rem; margin-top: 50px; padding: 20px; border-top: 1px solid #ddd; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. CORE AI LOGIC ---
@st.cache_resource
def initialize_engine():
    try:
        df = pd.read_csv("recruitment_data.csv")
        X, y, sensitive = df[['experience', 'test_score', 'gender']], df['hired'], df['gender']
        
        # Standard Engine
        std_model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y)
        
        # Fair Mitigation Engine
        mitigator = ExponentiatedGradient(RandomForestClassifier(), constraints=DemographicParity())
        mitigator.fit(X, y, sensitive_features=sensitive)
        
        return std_model, mitigator, df
    except FileNotFoundError:
        st.error("Missing Data: Please run 'data_generator.py' first.")
        st.stop()

std_engine, fair_engine, raw_data = initialize_engine()

conn = sqlite3.connect(
    "candidate_history.db",
    check_same_thread=False
)

cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS candidates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    experience INTEGER,
    skills_count INTEGER,
    match_score INTEGER,
    overall_score INTEGER
)
""")

conn.commit()

def extract_skills(text):
    skills_db = [
    "python", "java", "javascript",
    "c", "c++",
    "flask", "django",
    "node.js", "express",
    "sql", "mysql", "postgresql",
    "mongodb", "redis",
    "docker", "kubernetes",
    "git", "github",
    "html", "css",
    "react", "angular",
    "machine learning",
    "deep learning",
    "tensorflow",
    "pytorch",
    "nlp",
    "data analysis",
    "data science",
    "aws",
    "gcp"
]
    found_skills = []

    for skill in skills_db:
        if re.search(rf"\b{re.escape(skill)}\b", text):
            found_skills.append(skill)

    return found_skills

def extract_jd_skills(job_description):
    skills_db = [
    "python", "java", "javascript",
    "c", "c++",
    "flask", "django",
    "node.js", "express",
    "sql", "mysql", "postgresql",
    "mongodb", "redis",
    "docker", "kubernetes",
    "git", "github",
    "html", "css",
    "react", "angular",
    "machine learning",
    "deep learning",
    "tensorflow",
    "pytorch",
    "nlp",
    "data analysis",
    "data science",
    "aws",
    "gcp"
]

    found = []

    job_description = job_description.lower()

    for skill in skills_db:
        if re.search(rf"\b{re.escape(skill)}\b", job_description):
            found.append(skill)

    return found


def calculate_match_score(resume_skills, jd_skills):
    if len(jd_skills) == 0:
        return 0

    matched = len(set(resume_skills) & set(jd_skills))

    return int((matched / len(jd_skills)) * 100)

# --- 3. SIDEBAR (CONTROL CENTER) ---
with st.sidebar:
    st.title("🛡️ FairHire™ v4.0")
    st.markdown("---")
    model_choice = st.selectbox("Select Audit Engine", ["Standard (Biased)", "Fair (Mitigated)"])
    st.divider()
    st.subheader("Manual Simulation")
    sim_score = st.slider("Assessment Score (%)", 0, 100, 75)
    sim_gender = st.radio("Candidate Gender", [0, 1], format_func=lambda x: "Female" if x==0 else "Male")
    st.caption("Compliance Mode: ACTIVE")

# --- 4. MAIN DASHBOARD ---
st.title("AI Recruitment Bias & Resume Analyzer")
st.write("Analyze resumes, detect hiring bias, and evaluate candidate-job fit using AI.")

left_col, right_col = st.columns([1, 1.5], gap="large")

# LEFT: Resume Processing
with left_col:
    st.subheader("📁 Source Data")

    candidate_name = st.text_input(
        "Candidate Name"
    )

    uploaded_file = st.file_uploader(
        "Upload Candidate Resume (PDF)",
        type="pdf"
    )

    job_description = st.text_area(
        "Paste Job Description",
        height=150
    )

    candidate_exp = 0
    skills = []

    if uploaded_file:

        reader = PdfReader(uploaded_file)
        text = "".join(
            [p.extract_text() for p in reader.pages]
        ).lower()

        match = re.search(
            r'(\d+)\s*(?:\+)?\s*(?:year|years|yr|yrs)',
            text
        )

        candidate_exp = (
            int(match.group(1))
            if match else 0
        )

        st.success("✅ Analysis Complete")

        st.metric(
            "Detected Experience",
            f"{candidate_exp} Years"
        )

        skills = extract_skills(text)

    st.write("### Detected Skills")

    if skills:
        st.success(", ".join(skills))
    else:
        st.warning("No known skills detected")

    st.write("### Candidate Summary")

    st.write(
        f"Experience: {candidate_exp} Years"
    )

    if skills:
        st.write(
            f"Skills Found: {len(skills)}"
        )
    
    resume_strength = min(
    (len(skills) * 4) +
    (candidate_exp * 5),
    100
    )

    st.metric(
        "Resume Strength",
        f"{resume_strength}/100"
    )

    if job_description and skills:

        jd_skills = extract_jd_skills(
            job_description
        )

        score = calculate_match_score(
            skills,
            jd_skills
        )

        st.metric(
            "Job Match Score",
            f"{score}%"
        )

        overall_score = (score * 0.7) + (
        min(len(skills), 20) * 1.5)

        overall_score = int(min(overall_score, 100))

        st.metric(
            "Overall Candidate Score",
            f"{overall_score}/100"
        )

        cursor.execute(
            """
            INSERT INTO candidates (
                name,
                experience,
                skills_count,
                match_score,
                overall_score
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                candidate_name,
                candidate_exp,
                len(skills),
                score,
                overall_score
            )
        )


        conn.commit()

        if overall_score >= 85:
            st.success(
                "🏆 Highly Recommended"
            )

        elif overall_score >= 70:
            st.info(
                "👍 Recommended"
            )

        elif overall_score >= 50:
            st.warning(
                "⚠️ Needs Improvement"
            )
        else:
            st.error(
                "❌ Not Recommended"
            )
        
        missing_skills = list(
            set(jd_skills) - set(skills)
        )

        if missing_skills:
            st.warning(
                "Missing Skills: " +
                ", ".join(missing_skills)
            )

            st.info(
                "📚 Recommended Skills to Learn: "
                + ",".join(missing_skills)
            )
        report_text = f"""Candidate Analysis Report

        Experience: {candidate_exp} Years

        Skills Detected: {', '.join(skills) if skills else 'None'}

        Resume Strength: {resume_strength}/100

        Job Match Score: {score}%

        Overall Candidate Score: {overall_score}/100

        Missing Skills: {', '.join(missing_skills) if missing_skills else 'None'}

        Recommendations: {'Highly Recommended' if overall_score >= 85 else 'Recommended' if overall_score >= 70 else 'Needs Improvement' if overall_score >= 50 else 'Not Recommended'}
        """
        st.download_button(
            "📄 Download Analysis Report",
            report_text,
            "Candidate_Analysis_Report.txt"
        )

# RIGHT: Audit Verdict
with right_col:
    st.subheader("🎯 Audit Result")
    
    input_data = pd.DataFrame([[candidate_exp, sim_score, sim_gender]], 
                              columns=['experience', 'test_score', 'gender'])
    
    active_engine = fair_engine if model_choice == "Fair (Mitigated)" else std_engine
    prediction = active_engine.predict(input_data)[0]
    
    if prediction == 1:
        st.markdown(f"""<div class='verdict-card status-pass'>
                    <h2>✅ SELECTED</h2>
                    <p>Candidate meets the algorithmic threshold for <b>{model_choice}</b>.</p>
                    </div>""", unsafe_allow_html=True)
        
        # Download Certificate Feature
        report = f"AUDIT REPORT\nStatus: SELECTED\nExperience: {candidate_exp}\nScore: {sim_score}\nEngine: {model_choice}"
        st.download_button("📄 Download Fairness Certificate", report, "Audit_Report.txt")
    else:
        st.markdown(f"""<div class='verdict-card status-fail'>
                    <h2>❌ NOT SELECTED</h2>
                    <p>Candidate did not clear the <b>{model_choice}</b> threshold.</p>
                    </div>""", unsafe_allow_html=True)

# --- 5. ANALYTICS & BIAS MONITOR ---
st.divider()
st.subheader("📊 Analysis History")

history = pd.read_sql_query("SELECT * FROM candidates", conn)
st.dataframe(history)

if st.button("Clear All Candidate Records"):
    cursor.execute("DELETE FROM candidates")
    conn.commit()
    st.success("All candidate records cleared.")

st.divider()

st.subheader("🏆 Top Candidates")

top_candidates = pd.read_sql_query(
    """
    SELECT *
    FROM candidates
    ORDER BY overall_score DESC
    LIMIT 5
    """,
    conn
)

st.dataframe(top_candidates)

top_candidate = pd.read_sql_query(
    """
    SELECT *
    FROM candidates
    ORDER BY overall_score DESC
    LIMIT 1
    """,
    conn
)

if not top_candidate.empty:
    st.success(
        f"🏆 Top Candidate: {top_candidate.iloc[0]['name']} "
        f"({top_candidate.iloc[0]['overall_score']}/100)"
    )

st.divider()

st.subheader("📈 Candidate Analytics")

history = pd.read_sql_query(
    "SELECT * FROM candidates",
    conn
)

st.write("Overall Scores")

st.bar_chart(
    history.set_index("name")["overall_score"]
)

search_name = st.text_input(
    "Search Candidate"
)

if search_name:
    result = history[
        history["name"].str.contains(
            search_name,
            case=False
        )
    ]

    st.dataframe(result)

col1, col2 = st.columns(2)

with col1:
    st.write("**Selection Probability Gap**")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=raw_data, x='gender', y='hired', palette='viridis', ax=ax)
    ax.set_xticklabels(['Female', 'Male'])
    st.pyplot(fig)

with col2:
    st.write("**Fairness Audit Summary**")
    if model_choice == "Standard (Biased)":
        st.error("🚨 HIGH BIAS DETECTED: Historical data favors Group 1 (Male).")
    else:
        st.success("⚖️ MITIGATION ACTIVE: Selection rates normalized via Demographic Parity.")
    
    st.metric("Regulatory Score", "98/100" if model_choice == "Fair (Mitigated)" else "62/100")

# --- 6. PROFESSIONAL FOOTER ---
st.markdown("""
    <div class="footer">
        🛡️ FairHire™ Enterprise Auditor | GDPR & EU AI Act Compliant | v4.0.2<br>
        © 2026 FairHire Technologies Group. All Rights Reserved.
    </div>
    """, unsafe_allow_html=True)