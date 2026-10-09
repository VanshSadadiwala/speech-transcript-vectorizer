# Speech Transcript Vectorization using Bag-of-Words and Collocation Features

**Student:** ET23BTCO127  |  **Problem 10**  |  MT-AIML, Module 3 (Word Representations & Feature Extraction)

Converts raw speech-to-text transcripts (voice-assistant queries) into a Bag-of-Words matrix
**augmented with PMI-selected collocation columns**, so that a phrase such as `credit card`
becomes its own feature (`credit_card`). Includes a Streamlit web app, an evaluation pipeline,
tests and the two required reports.

## 1. Run the web app (easiest)

Requires Python 3.9 or newer and an internet connection for the first run (to install packages).

* **Windows:** double-click `run_app.bat`
* **macOS / Linux:** `./run_app.sh`

The script creates a virtual environment, installs `requirements.txt` and opens the app in your browser
(normally http://localhost:8501).

Manual alternative:

```bash
pip install -r requirements.txt
streamlit run app.py
```

### Using the app
1. **Input** (sidebar): sample dataset (350 labelled queries), paste your own transcripts (one per line) or upload `.txt` / `.csv`.
2. **Preprocessing**: tick the cleaning steps.
3. **Features**: set BoW `min_df` / `max_features`, collocation type, **minimum frequency**, **PMI threshold**.
4. Press **Run pipeline**. Tabs show: cleaning preview, feature matrix (collocation columns highlighted, CSV download),
   collocation table + chart, dimensionality statistics and threshold curve, and an intent-classification demo.

## 2. Run the evaluation pipeline (no browser)

```bash
python -m src.pipeline
```

Writes CSV tables and PNG charts to `results/` (preprocessing statistics, configuration comparison,
PMI-threshold sweep, top collocations, sample feature matrix). Use your own data with
`python -m src.pipeline --data my.csv --out results` (CSV with columns `text` and optionally `intent`).

## 3. Run the tests

```bash
python -m unittest discover -s tests -v
```

## Project structure

```
.
├── app.py                         Streamlit web application
├── run_app.bat / run_app.sh       one-click launchers
├── requirements.txt
├── docs/
│   ├── Architecture_and_Methodology.pdf   formulas, design, feature-matrix, PMI thresholding, dimensionality
│   └── Evaluation_Report.pdf              results, tables, charts, failure analysis
├── src/
│   ├── preprocessing.py           noise/filler removal, contractions, tokenization, stop words, lemmatization
│   ├── models.py                  CollocationExtractor (PMI) + AugmentedBoWVectorizer
│   └── pipeline.py                evaluation runner (comparisons, threshold sweep, charts)
├── data/
│   ├── raw_transcripts/voice_queries.txt  raw ASR-style transcripts
│   └── annotated/intents.csv              transcripts with intent labels
├── scripts/make_dataset.py        regenerates the synthetic dataset (seed 42)
├── tests/test_core.py             unit tests
└── results/                       output of src.pipeline
```

## Key formula

`PMI(x, y) = log2( P(x, y) / ( P(x) * P(y) ) )`, estimated with `nltk.collocations`.
PMI over-rewards rare pairs, so a minimum-frequency filter (default 5) is applied before scoring.

## Notes

* The dataset is **synthetic** (templates + simulated ASR noise). It demonstrates the pipeline; it is not real-world evidence.
* NLTK stop-word and WordNet data are downloaded automatically the first time. If the download is impossible (offline),
  the code falls back to a built-in stop-word list and a simple suffix lemmatizer.
* Everything is deterministic (seed 42).
