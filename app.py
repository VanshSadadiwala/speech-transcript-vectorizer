"""Streamlit web app: Speech Transcript Vectorizer (BoW + PMI collocation features).

Run:  streamlit run app.py
"""
import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.models import AugmentedBoWVectorizer, CollocationExtractor
from src.pipeline import DEFAULT_DATA, load_dataset
from src.preprocessing import PreprocessConfig, TranscriptPreprocessor, preprocessing_stats

st.set_page_config(page_title="Transcript Vectorizer", page_icon="🎙️", layout="wide")
st.title("🎙️ Speech Transcript Vectorizer")
st.caption("Bag-of-Words + PMI collocation features  |  Problem 10  |  ET23BTCO127")

SAMPLE = """Hey um I'm gonna need you to turn off the air conditioning
Can you play jazz music
CALL my credit card company
uh set an alarm for six thirty
okay turn up the volume
Could you please call the credit card helpline
navigate to the airport you know
play the next song
turn on the air conditioning please
turn down the volume
call customer service
Set a timer for 10 minutes
call customer service please
play some jazz music"""

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("1. Input")
    source = st.radio("Transcript source", ["Sample dataset (350 queries)", "Paste text", "Upload file"])
    labels = None
    if source.startswith("Sample"):
        df_in = load_dataset(DEFAULT_DATA)
        texts = df_in["text"].tolist()
        labels = df_in["intent"].tolist() if "intent" in df_in else None
        st.caption(f"{len(texts)} labelled in-car/voice-assistant queries loaded.")
    elif source == "Paste text":
        raw = st.text_area("One transcript per line", SAMPLE, height=220)
        texts = [l.strip() for l in raw.splitlines() if l.strip()]
    else:
        up = st.file_uploader("Upload .txt (one per line) or .csv (column `text`)", type=["txt", "csv"])
        texts = []
        if up is not None:
            if up.name.lower().endswith(".csv"):
                d = pd.read_csv(up)
                col = "text" if "text" in d.columns else d.columns[0]
                texts = d[col].dropna().astype(str).tolist()
                labels = d["intent"].tolist() if "intent" in d.columns else None
            else:
                texts = [l.strip() for l in io.StringIO(up.getvalue().decode("utf-8", "ignore")) if l.strip()]

    st.header("2. Preprocessing")
    c_fill = st.checkbox("Remove fillers (um, uh, you know)", True)
    c_contr = st.checkbox("Expand contractions (gonna, what's)", True)
    c_stop = st.checkbox("Remove stop words (keeps on/off/up/down)", True)
    c_lem = st.checkbox("Lemmatize", True)
    c_num = st.checkbox("Normalise numbers to 'num'", True)

    st.header("3. Features")
    min_df = st.number_input("BoW min document frequency (min_df)", 1, 20, 1)
    max_feat = st.number_input("BoW max features (0 = no limit)", 0, 5000, 0, step=50)
    use_col = st.checkbox("Add collocation features", True)
    ngram_opt = st.selectbox("Collocation type", ["Bigrams", "Trigrams", "Bigrams + Trigrams"])
    default_mf = 5 if len(texts) >= 100 else 2   # small inputs need a lower cut-off
    min_freq = st.slider("Collocation min frequency", 1, 20, default_mf, key=f"mf_{default_mf}")
    thr = st.slider("PMI threshold (bits)", -2.0, 12.0, 3.0, 0.5)
    top_cap = st.number_input("Max collocation columns (0 = no limit)", 0, 2000, 0, step=10)
    binary = st.checkbox("Binary presence instead of counts", False)
    run = st.button("Run pipeline", type="primary", width="stretch")

if not texts:
    st.info("Provide some transcripts in the sidebar, then press **Run pipeline**.")
    st.stop()
if not run and "ran_once" not in st.session_state:
    st.info(f"{len(texts)} transcripts ready. Press **Run pipeline** in the sidebar.")
    st.stop()
st.session_state["ran_once"] = True

# ---------------------------------------------------------------- compute
cfg = PreprocessConfig(remove_fillers=c_fill, expand_contractions=c_contr, remove_stopwords=c_stop,
                       lemmatize=c_lem, normalize_numbers=c_num)
pre = TranscriptPreprocessor(cfg)
docs = pre.process_many(texts)
sizes = {"Bigrams": (2,), "Trigrams": (3,), "Bigrams + Trigrams": (2, 3)}[ngram_opt]

vec = AugmentedBoWVectorizer(min_df=int(min_df), max_features=int(max_feat) or None, use_collocations=use_col,
                             min_freq=min_freq, pmi_threshold=thr, ngram_sizes=sizes,
                             max_collocations=int(top_cap) or None, binary=binary)
X = vec.fit_transform(docs)
stats = vec.stats(X)
pstats = preprocessing_stats(texts, docs)

if use_col and not vec.collocation_feature_names_:
    st.warning("No collocations passed the filters. Lower the PMI threshold or the minimum frequency, or add more transcripts.")

# ---------------------------------------------------------------- headline numbers
m = st.columns(5)
m[0].metric("Transcripts", stats["documents"])
m[1].metric("BoW columns", stats["bow_features"])
m[2].metric("Collocation columns", stats["collocation_features"])
m[3].metric("Total features", stats["total_features"])
m[4].metric("Sparsity", f"{stats['sparsity_pct']}%")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Cleaning preview", "Feature matrix", "Collocations", "Dimensionality", "Intent demo"])

with tab1:
    st.subheader("Raw transcript vs cleaned tokens")
    prev = pd.DataFrame({"raw": texts, "cleaned tokens": [" ".join(d) for d in docs]})
    st.dataframe(prev.head(200), width="stretch", height=320)
    st.write("**Preprocessing statistics**")
    st.dataframe(pd.DataFrame([pstats]), width="stretch")

with tab2:
    st.subheader("Augmented feature matrix  =  [ BoW columns | collocation columns ]")
    n_show = st.slider("Rows to display", 5, min(100, len(texts)), min(20, len(texts)))
    dfm = vec.to_dataframe(X[:n_show], index=[f"doc{i + 1}" for i in range(n_show)])
    ncol = len(vec.collocation_feature_names_)
    col_names = set(vec.collocation_feature_names_)
    only_col = st.checkbox("Show only collocation columns", False)
    if only_col and ncol:
        dfm = dfm[[c for c in dfm.columns if c in col_names]]
    st.dataframe(dfm.style.apply(lambda s: ["background-color:#DDF3EF" if s.name in col_names else "" for _ in s], axis=0),
                 width="stretch", height=360)
    st.caption("Green columns are collocation features; the others are single-word BoW features.")
    full = vec.to_dataframe(X)
    st.download_button("⬇ Download full feature matrix (CSV)", full.to_csv(index=False).encode(), "feature_matrix.csv", "text/csv")

with tab3:
    if not use_col or vec.collocations_ is None:
        st.warning("Collocation features are switched off.")
    else:
        ce = vec.collocations_
        sel = ce.selected_df_
        st.subheader(f"{len(sel)} collocations selected (min_freq ≥ {min_freq}, PMI ≥ {thr})")
        st.dataframe(sel, width="stretch", height=300)
        if len(sel):
            st.bar_chart(sel.head(20).set_index("phrase")["pmi"])
        st.download_button("⬇ Download collocation table (CSV)", sel.to_csv(index=False).encode(), "collocations.csv", "text/csv")
        st.info("PMI favours rare pairs. A pair that appears only twice can get a huge PMI by chance, "
                "so keep the minimum frequency filter at 3 or more.")

with tab4:
    st.subheader("Vector dimensionality management")
    st.json(stats)
    if use_col and vec.collocations_ is not None:
        scored = vec.collocations_.scored_
        sweep_t = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8]
        n_bow = stats["bow_features"]
        sw = pd.DataFrame({"PMI threshold": sweep_t,
                           "total features": [n_bow + int((scored["pmi"] >= t).sum()) for t in sweep_t]})
        st.write("**Effect of the PMI threshold on matrix width** (current min_freq and collocation type)")
        st.line_chart(sw.set_index("PMI threshold"))
        st.dataframe(sw.T, width="stretch")

with tab5:
    if labels is None:
        st.info("Intent demo needs labelled data (CSV with `text` and `intent`). Choose the sample dataset or upload such a CSV.")
    else:
        from sklearn.metrics import accuracy_score, classification_report
        from sklearn.model_selection import train_test_split
        from sklearn.svm import LinearSVC
        st.subheader("Does the feature matrix recognise intents?")
        test_size = st.slider("Test fraction", 0.1, 0.8, 0.3, 0.05)
        idx = np.arange(len(docs))
        tr, te = train_test_split(idx, test_size=test_size, random_state=42, stratify=labels)
        results = []
        for name, use in (("BoW only", False), ("BoW + collocations", True)):
            v = AugmentedBoWVectorizer(min_df=int(min_df), max_features=int(max_feat) or None, use_collocations=use,
                                       min_freq=min_freq, pmi_threshold=thr, ngram_sizes=sizes,
                                       max_collocations=int(top_cap) or None, binary=binary)
            Xtr = v.fit_transform([docs[i] for i in tr])
            Xte = v.transform([docs[i] for i in te])
            clf = LinearSVC(random_state=42).fit(Xtr, [labels[i] for i in tr])
            pred = clf.predict(Xte)
            results.append({"features": name, "columns": Xtr.shape[1],
                            "accuracy": round(accuracy_score([labels[i] for i in te], pred), 4)})
        st.dataframe(pd.DataFrame(results), width="stretch")
        st.caption("PMI is learned from the training split only, so there is no leakage into the test split.")
