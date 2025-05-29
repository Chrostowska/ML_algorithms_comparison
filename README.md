# Machine Learning Algorithms Comparison

A comprehensive machine learning pipeline for comparing different classification algorithms on the Breast Cancer Wisconsin dataset. This project implements various machine learning models and provides tools for model evaluation and comparison.

## Project Structure

```
ml_project/
├── ml_pipeline/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── models.py
│   └── visualization/
│       └── __init__.py
├── data/              # Dataset storage
├── notebooks/         # Jupyter notebooks for analysis
├── tests/            # Unit tests
├── outputs/          # Model outputs and visualizations
├── main.py           # Main execution script
├── requirements.txt  # Project dependencies
└── README.md
```

## Features

- **Data Preprocessing**
  - Feature scaling and normalization
  - Train-test splitting
  - Data validation and cleaning

- **Model Implementation**
  - Logistic Regression
  - Random Forest
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors (KNN)
  - Decision Tree

- **Model Evaluation**
  - Accuracy metrics
  - Cross-validation
  - Classification reports
  - Confusion matrices
  - ROC curves

- **Visualization**
  - Performance metric plots
  - Confusion matrix visualization
  - ROC curve plotting

## Results

The project evaluates multiple classification algorithms on the Breast Cancer Wisconsin dataset:

- SVM: Best performing model with 98.25% test accuracy
- Logistic Regression: 97.37% test accuracy
- Random Forest: 96.49% test accuracy
- KNN and Decision Tree: 94.74% test accuracy

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Chrostowska/ML_algorithms_comparison.git
cd ml_project
```

2. Create and activate virtual environment:
```bash
python -m venv venv
.\venv\Scripts\activate  # On Windows
source venv/bin/activate # On Unix/MacOS
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main script to execute the complete pipeline:
```bash
python main.py
```

This will:
1. Load and preprocess the Breast Cancer Wisconsin dataset
2. Train multiple classification models
3. Evaluate and compare model performance
4. Generate visualization outputs

## Dependencies

- numpy>=1.21.0
- pandas>=1.3.0
- scikit-learn>=1.0.0
- matplotlib>=3.4.0
- seaborn>=0.11.0
- xgboost>=1.5.0
- pytest>=7.0.0

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 