# AAI-500 SF Health Inspection

This repository contains a data science and machine learning project focused on San Francisco health inspections. The goal is to predict whether a food establishment will pass or fail an inspection and to rank upcoming inspections by risk so that higher-risk facilities can be prioritized.

## Project Overview

The workflow includes:

- Loading and cleaning raw inspection data
- Engineering categorical features relevant to inspection outcomes
- Training and comparing baseline classification models
- Evaluating model performance using precision, recall, and average precision
- Saving the best-performing model
- Generating a risk-priority list for simulated upcoming inspections

This project is structured around a reproducible pipeline in the `src/` directory and uses a cleaned dataset saved under `data/`.

## Repository Structure

```text
AAI-500-sf-health-Inspection/
├── data/
│   ├── cleaned-data/
│   │   ├── cleaned_data_eda.csv
│   │   └── cleaned_data_modeling.csv
│   ├── raw-data/
│   │   └── export.csv
│   └── outputs/
│       └── (supporting outputs if generated)
├── figures/
│   └── feature_importance.png
├── outputs/
│   └── inspection_priorities.csv
├── src/
│   ├── data_cleaning.py
│   ├── eda.py
│   ├── model_training.py
│   └── predict.py
├── .gitignore
├── requirements.txt
├── README.md
└── models/
    └── best_model.pkl
```

## Dataset

The project uses San Francisco health inspection data stored in the `data/raw-data/` folder. The data is filtered to remove rows missing the `facility_rating_status` field and transformed into a binary target:

- `0` = Pass
- `1` = Fail

Key features include:

- `permit_type`
- `analysis_neighborhood`
- `inspection_frequency_type`
- `inspection_type`

The cleaning script converts these features into a modeling-ready format and exports cleaned datasets for both exploratory analysis and model training.

## Modeling Approach

The training script compares several classifiers:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

Models are evaluated with stratified cross-validation and assessed using:

- Recall
- Precision
- Average Precision (AP)

The best model is selected based on the cross-validation average precision score and saved to `models/best_model.pkl`.

## How to Run

1. Create and activate a virtual environment (optional but recommended):

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the data cleaning pipeline:

```bash
python src/data_cleaning.py
```

4. Train the models:

```bash
python src/model_training.py
```

5. Generate inspection priority predictions:

```bash
python src/predict.py
```

## Outputs

The project produces the following artifacts:

- `data/cleaned-data/cleaned_data_modeling.csv` — modeling-ready dataset
- `data/cleaned-data/cleaned_data_eda.csv` — exploratory analysis dataset
- `figures/feature_importance.png` — feature importance visualization
- `outputs/inspection_priorities.csv` — simulated inspection priority rankings
- `models/best_model.pkl` — trained classification model
- `figures/` - contains 6 EDA visualizations

## Dependencies

The project relies on Python libraries such as:

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- joblib

See `requirements.txt` for the full dependency list.

## License

This project does not currently include a formal license file. Please check with the repository owner before reuse in a production or public-facing context.

## Notes

This project is intended as a practical machine learning example for risk-based inspection prioritization. It demonstrates a full pipeline from raw data processing to model training, evaluation, and decision support output.
