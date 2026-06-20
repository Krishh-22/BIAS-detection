# 🛡️ FairHire™ Enterprise AI Auditor

### Overview
An automated recruitment auditing suite designed to detect and mitigate algorithmic bias using the **Fairlearn** framework. This tool ensures compliance with global AI ethics standards.

### Features
- **PDF Resume Parsing:** Automated extraction of candidate metrics.
- **Bias Mitigation:** Toggle between Standard (Biased) and Fair (Mitigated) AI engines.
- **Fairness Certification:** Downloadable compliance reports for legal record-keeping.

### Setup
1. `python -m venv venv`
2. `.\venv\Scripts\activate`
3. `pip install -r requirements.txt`
4. `python data_generator.py`
5. `python -m streamlit run app.py`