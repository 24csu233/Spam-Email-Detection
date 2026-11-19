# Machine Learning-Based Spam Email Detection

A comparative machine-learning project using **Naive Bayes, Logistic Regression, SVM, and Random Forest**.

## Important dataset note
`data/spam.csv` contains a **small demonstration dataset** so the project runs immediately. It is not large or representative enough for final research conclusions. For your final experiments, replace it with a larger, properly cited email dataset (for example, the SpamAssassin public corpus) formatted as a CSV with columns:
- `text`: email subject/body text
- `label`: `spam` or `ham` (not spam)

Do not report demo-dataset scores as research findings.

## Setup (Windows / VS Code)
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run
```powershell
python train.py
```
The script preprocesses text, trains four models, prints metrics, and saves:
- `results/model_comparison.csv`
- `results/metrics_comparison.png`
- `results/confusion_matrix_*.png`
- trained models in `models/`

Optional web demo:
```powershell
streamlit run app.py
```
Train the models first. The app loads the saved models and predicts whether entered text is spam or ham.

## Dataset format
Example:
```csv
text,label
"Congratulations, you won a prize. Click here",spam
"Please find the meeting notes attached",ham
```

## Methodology
1. Load and validate labeled email text.
2. Split into training and test sets using stratification.
3. Convert text to TF-IDF features.
4. Train four classifiers using a shared preprocessing pipeline.
5. Compare accuracy, precision, recall, F1-score, and confusion matrices.

## Research integrity
Use a sufficiently large dataset, cite its original source, record the dataset version and split/random seed, and rerun experiments before writing results in your paper. Performance can vary by dataset and split.
