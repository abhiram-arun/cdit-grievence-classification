import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle

# Load Excel
df = pd.read_excel("Translated_final (1).xlsx")

# Load sentence transformer
model_st = SentenceTransformer("all-MiniLM-L6-v2")

# Ensure NaNs are handled
df["department"] = df["department"].fillna("")
df["subjectNameEng"] = df["subjectNameEng"].fillna("")
df["microSubjectNameEng"] = df.get("microSubjectNameEng", "").fillna("")

# --- DEPARTMENT EMBEDDINGS ---
dept_texts = df["department"].dropna().unique().tolist()
dept_embeddings = model_st.encode(dept_texts, convert_to_numpy=True)

# --- SUBJECT EMBEDDINGS with Department Context ---
subject_rows = df[["department", "subjectNameEng"]].dropna().drop_duplicates()
subject_texts = (subject_rows["department"] + " " + subject_rows["subjectNameEng"]).tolist()
subject_embeddings = model_st.encode(subject_texts, convert_to_numpy=True)

# --- MICRO-SUBJECT EMBEDDINGS with Department + Subject Context ---
if "microSubjectNameEng" in df.columns:
    micro_rows = df[["department", "subjectNameEng", "microSubjectNameEng"]].dropna().drop_duplicates()
    micro_texts = (
        micro_rows["department"] + " " +
        micro_rows["subjectNameEng"] + " " +
        micro_rows["microSubjectNameEng"]
    ).tolist()
    micro_embeddings = model_st.encode(micro_texts, convert_to_numpy=True)
else:
    micro_texts, micro_embeddings = [], []

# Save all
with open("encodings.pkl", "wb") as f:
    pickle.dump({
        "dept_texts": dept_texts,
        "dept_embeddings": dept_embeddings,
        "subject_rows": subject_rows,
        "subject_texts": subject_texts,
        "subject_embeddings": subject_embeddings,
        "micro_rows": micro_rows if micro_texts else [],
        "micro_texts": micro_texts,
        "micro_embeddings": micro_embeddings
    }, f)

print("Saved contextual embeddings.")