import pandas as pd
import torch
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from IndicTransToolkit.processor import IndicProcessor
import spacy

# Load models
st_model = SentenceTransformer("all-MiniLM-L6-v2")
translator_model = AutoModelForSeq2SeqLM.from_pretrained("ai4bharat/indictrans2-indic-en-1B", trust_remote_code=True).to("cuda" if torch.cuda.is_available() else "cpu")
translator_tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indictrans2-indic-en-1B", trust_remote_code=True)
ip = IndicProcessor(inference=True)

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


# Load embeddings
with open("encodings.pkl", "rb") as f:
    data = pickle.load(f)

def preprocess(text):
    doc = nlp(text)
    stop_words = nlp.Defaults.stop_words
    tokens = [
        token.text.lower() for token in doc
        if token.text.lower() not in stop_words and not token.is_punct and not token.is_space
    ]
    return " ".join(tokens)

def translate_malayalam(sentence):
    batch = ip.preprocess_batch([sentence], src_lang="mal_Mlym", tgt_lang="eng_Latn")
    inputs = translator_tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(translator_model.device)
    with torch.no_grad():
        output = translator_model.generate(**inputs, max_length=256)
    return translator_tokenizer.batch_decode(output, skip_special_tokens=True)[0]

def classify(text):
    # Translate if Malayalam
    if any("\u0D00" <= c <= "\u0D7F" for c in text):  # Unicode range for Malayalam
        text = translate_malayalam(text)
        print(f"Translated: {text}")
    
    text = preprocess(text)
    embedding = st_model.encode(text, convert_to_numpy=True)

    # --- Department ---
    dept_sim = cosine_similarity([embedding], data["dept_embeddings"])[0]
    best_dept_idx = dept_sim.argmax()
    best_dept = data["dept_texts"][best_dept_idx]
    print(f"\n Department: {best_dept} (score: {dept_sim[best_dept_idx]:.4f})")

    # --- Subject ---
    subject_filtered = data["subject_rows"][data["subject_rows"]["department"] == best_dept]
    if subject_filtered.empty:
        return

    subject_texts = (subject_filtered["department"] + " " + subject_filtered["subjectNameEng"]).tolist()
    subject_embeddings = st_model.encode([preprocess(t) for t in subject_texts], convert_to_numpy=True)
    subj_sim = cosine_similarity([embedding], subject_embeddings)[0]
    best_subj_idx = subj_sim.argmax()
    best_subject = subject_filtered.iloc[best_subj_idx]["subjectNameEng"]
    print(f" Subject: {best_subject} (score: {subj_sim[best_subj_idx]:.4f})")

    # --- Micro-Subject ---
    if isinstance(data["micro_rows"], pd.DataFrame) and not data["micro_rows"].empty:
        micro_filtered = data["micro_rows"][
            (data["micro_rows"]["department"] == best_dept) &
            (data["micro_rows"]["subjectNameEng"] == best_subject)
        ]
        if not micro_filtered.empty:
            micro_texts = (
                micro_filtered["department"] + " " +
                micro_filtered["subjectNameEng"] + " " +
                micro_filtered["microSubjectNameEng"]
            ).tolist()
            micro_embeddings = st_model.encode([preprocess(t) for t in micro_texts], convert_to_numpy=True)
            micro_sim = cosine_similarity([embedding], micro_embeddings)[0]
            best_micro_idx = micro_sim.argmax()
            best_micro = micro_filtered.iloc[best_micro_idx]["microSubjectNameEng"]
            print(f" Micro-Subject: {best_micro} (score: {micro_sim[best_micro_idx]:.4f})")


# Example usage:

classify("പുന്നപ്ര കാച്ചിൽമുക്ക് പ്രദേശത്തെ കുടിവെള്ള ക്ഷാമം ദൈനംദിന ജീവിതത്തെ ബാധിക്കുന്നു.")