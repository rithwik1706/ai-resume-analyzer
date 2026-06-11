# AI Resume Analyzer — RAG-Based Resume Intelligence Platform

## Overview

* This project is built using **RAG (Retrieval-Augmented Generation)** and **LLM-powered NLP techniques**
* Supports **contextual resume understanding** and **conversational resume Q&A**
* Uses **Sentence Transformers + ChromaDB** for semantic retrieval
* Provides **ATS scoring, job-description matching, and skill-gap analysis**
* Deployed using **Streamlit Cloud**

---

## Live Preview

* Open the deployed app:

  * https://ai-resume-analyzer-ma2rbxzinds54ggha4fpv3.streamlit.app/

---

## Requirements

* Python 3.10+
* pip
* Git

---

## Required Libraries

Install all dependencies:

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install streamlit langchain chromadb sentence-transformers mistralai pypdf python-dotenv
```

---

## Project Structure

```text
ai-resume-analyzer/
│
├── app.py
├── requirements.txt
├── .env
├── chroma_db/
├── resume_parser.py
├── ats_scoring.py
├── jd_matching.py
├── skill_gap_analysis.py
└── README.md
```

---

## Running Locally

```bash
streamlit run app.py
```

---

## Environment Variables

Create a `.env` file and add:

```env
MISTRAL_API_KEY=your_api_key_here
```

---

## Deployment (Streamlit Cloud)

* Push project to GitHub
* Connect GitHub repository to Streamlit Cloud
* Add required environment variables
* Deploy application

---

## Important Notes

* Ensure `.env` file is added to `.gitignore`
* Do not expose API keys publicly
* ChromaDB is used for efficient semantic vector retrieval
* Resume files are processed dynamically for contextual analysis

---

## Features

* Conversational Resume Q&A
* ATS Score Analysis
* Job Description Matching
* Skill Gap Analysis
* Semantic Search using Embeddings
* RAG-Based Contextual Reasoning
* PDF Resume Upload & Processing
* Real-Time Resume Analysis
