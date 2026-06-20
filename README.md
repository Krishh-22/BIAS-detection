# AI Recruitment Bias Detection & Resume Analyzer

## Overview

AI Recruitment Bias Detection & Resume Analyzer is an intelligent recruitment analytics platform built using Python, Streamlit, Scikit-Learn, Fairlearn, SQLite, and NLP techniques.

The system analyzes candidate resumes, extracts skills and experience, compares resumes against job descriptions, calculates candidate-job fit scores, detects missing skills, generates recommendations, and evaluates potential bias in recruitment decisions using fairness-aware machine learning.

This project demonstrates practical applications of Artificial Intelligence, Machine Learning, NLP, Data Analytics, and Responsible AI in modern recruitment workflows.

---

## Key Features

### Resume Analysis

* Upload PDF resumes
* Automatic experience extraction
* Automatic skill extraction using NLP and regex-based matching
* Resume strength evaluation

### Candidate-Job Matching

* Job Description (JD) analysis
* Skill matching against job requirements
* Job Match Score calculation
* Missing skill identification
* Personalized improvement recommendations

### Candidate Evaluation

* Resume Strength Score
* Overall Candidate Score
* Candidate recommendation engine
* Top candidate identification

### Fair AI & Bias Detection

* Standard recruitment model
* Fairness-aware recruitment model
* Bias detection dashboard
* Demographic parity analysis using Fairlearn

### Data Management

* SQLite database integration
* Candidate history tracking
* Search candidate functionality
* Candidate leaderboard

### Reporting & Analytics

* Downloadable candidate reports
* Historical candidate analytics
* Interactive dashboards
* Recruitment insights visualization

---

## Tech Stack

### Programming Language

* Python

### Frontend

* Streamlit

### Machine Learning

* Scikit-Learn
* Fairlearn

### Data Processing

* Pandas
* NumPy

### Visualization

* Matplotlib
* Seaborn

### Database

* SQLite

### PDF Processing

* PyPDF2

---

## System Workflow

1. Upload Candidate Resume (PDF)
2. Extract Experience & Skills
3. Parse Job Description
4. Calculate Job Match Score
5. Generate Candidate Evaluation Metrics
6. Detect Missing Skills
7. Provide Recommendations
8. Store Results in Database
9. Generate Reports & Analytics
10. Monitor Recruitment Fairness

---

## Project Highlights

* Built an end-to-end AI-powered recruitment analytics platform.
* Implemented resume parsing and skill extraction from PDF documents.
* Developed candidate-job matching algorithms for recruitment screening.
* Integrated fairness-aware machine learning techniques to identify hiring bias.
* Designed an analytics dashboard with candidate history, search, ranking, and reporting capabilities.
* Utilized SQLite for persistent candidate data storage and historical analysis.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Krishh-22/BIAS-detection.git
```

Navigate to the project directory:

```bash
cd BIAS-detection-main
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## Future Enhancements

* AI-powered resume summarization using LLMs
* Semantic skill matching using embeddings
* Multi-resume ranking system
* Recruiter dashboard
* Cloud deployment and authentication
* Advanced candidate recommendation engine

---

## Author

Krishna Nishad

B.Tech Computer Science Engineering Student

Interests:

* Artificial Intelligence
* Machine Learning
* Backend Development
* Software Engineering
* Responsible AI

---

## License

This project is developed for educational, research, and portfolio purposes.
