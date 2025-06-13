# IndicTrans2 Malayalam to English Translation

A Malayalam-English bilingual grievance classification system that categorizes Malayalam speaking citizen's inputs into:

    🏢 Department

    📂 Subject

    📄 Micro-Subject

using IndicTrans2 for translation and Sentence Transformers for semantic similarity.

This tool translates Malayalam "subject" and "micro-subject" columns in an Excel file into English using [AI4Bharat's IndicTrans2](https://github.com/AI4Bharat/IndicTrans2) and [IndicTransToolkit](https://github.com/VarunGumma/IndicTransToolkit).

## 📦 Requirements

- Python 3.8+
- pip
- CUDA GPU (optional, but recommended)

## 🔧 Setup

```bash
git clone https://github.com/yourusername/your-repo.git
cd your-repo
bash install.sh
```

## Usage

embeddings.py = Generates semantic embeddings for all department, subject, and micro-subject labels and stores them in encodings.pkl.

classify.py = Given a Malayalam or English input grievance, classifies it into:

    Most relevant Department

    Most relevant Subject (under that department)

    Most relevant Micro-subject (under that subject)

Input provided as a string in the last line 
            
    classify("Your required grievence")


