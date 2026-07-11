# OptiCrop – Smart Agricultural Production Optimization Engine

A Machine Learning-powered agricultural recommendation system that predicts the most suitable crop based on soil nutrients and environmental conditions. Built as part of the **SmartBridge Virtual Internship**.

---

## Project Overview

OptiCrop analyzes seven key agricultural parameters — Nitrogen (N), Phosphorous (P), Potassium (K), Temperature, Humidity, pH, and Rainfall — and recommends the best crop to plant for maximum yield and sustainability. The system also includes a crop suitability checker and an interactive research dashboard.

### Target Audience
- Farmers
- Agricultural Researchers
- Agribusiness Companies
- Policymakers

---

## Features

- **Scenario 1 – Crop Recommendation**: Enter soil and climate values; the system predicts the best crop with a confidence score, suitable conditions, benefits, farming practices, season, water requirement, fertilizer recommendation, and expected productivity.
- **Scenario 2 – Suitability Check**: Select a crop and enter conditions; the system computes a suitability score (0–100%), advantages, possible issues, and improvement suggestions.
- **Scenario 3 – Research Dashboard**: Interactive charts for crop distribution, NPK analysis, rainfall/temperature/humidity/pH distributions, and a feature correlation heatmap.
- **5 ML Models Compared**: Decision Tree, Random Forest, Logistic Regression, KNN, and K-Means Clustering. The best model (Random Forest, ~87% accuracy) is automatically selected.
- **Bonus Features**: Dark mode toggle, loading animation, animated prediction button, typing effect, scroll animations, back-to-top button, and fully responsive mobile design.

---

## Technologies Used

| Category          | Technologies                                      |
|-------------------|---------------------------------------------------|
| Backend           | Python 3.10+, Flask                               |
| Machine Learning  | Scikit-learn, NumPy, Pandas, SciPy, Pickle         |
| Frontend          | HTML5, CSS3, Bootstrap 5, JavaScript, Font Awesome|
| Visualization     | Matplotlib, Seaborn, Chart.js                     |
| Notebook          | Jupyter Notebook                                  |
| Version Control   | Git                                               |

---

## Project Structure

```
OptiCrop/
├── app.py                      # Flask application
├── requirements.txt            # Python dependencies
├── README.md                    # Project documentation
├── model/                       # Trained ML model artifacts
│   ├── crop_model.pkl           # Best model (Random Forest)
│   ├── scaler.pkl               # StandardScaler
│   └── label_encoder.pkl        # LabelEncoder
├── dataset/
│   └── Crop_recommendation.csv  # Training dataset (2,200 rows, 22 crops)
├── notebooks/
│   └── Crop_Model_Training.ipynb # EDA + model training notebook
├── templates/                   # Flask HTML templates
│   ├── base.html
│   ├── index.html
│   ├── result.html
│   ├── scenario2.html
│   ├── dashboard.html
│   ├── about.html
│   ├── contact.html
│   └── error.html
├── static/
│   ├── css/style.css            # Custom styles (glassmorphism, dark mode, animations)
│   ├── js/script.js             # Main JavaScript (dark mode, animations, validation)
│   ├── js/dashboard.js          # Dashboard Chart.js visualizations
│   └── images/
├── utils/                       # Python utility modules
│   ├── __init__.py
│   ├── preprocess.py            # Data loading, cleaning, scaling, encoding, validation
│   ├── train_model.py           # Model training pipeline
│   ├── evaluation.py            # Model evaluation metrics
│   ├── predict.py               # Inference helpers
│   ├── helper.py                # Crop knowledge base, suitability, dashboard analytics
│   └── generate_dataset.py      # Dataset generation script
└── screenshots/                 # Add your screenshots here
```

---

## Installation Steps

### 1. Clone the Repository

```bash
git clone <repository-url>
cd OptiCrop
```

### 2. Create a Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

---

## Model Training

### Option A: Using the Jupyter Notebook

```bash
cd notebooks
jupyter notebook Crop_Model_Training.ipynb
```

Run all cells from top to bottom. The notebook performs complete EDA, trains all 5 models, evaluates them, and saves the artifacts to `model/`.

### Option B: Using the Training Script

```bash
python utils/train_model.py
```

This trains all models, prints the comparison table, and saves the best model, scaler, and label encoder to the `model/` directory.

### Option C: Regenerate the Dataset

If you want to regenerate the synthetic dataset:

```bash
python utils/generate_dataset.py
```

---

## Run the Flask Server

```bash
python app.py
```

Open your browser and navigate to: **http://127.0.0.1:5000**

---

## Usage Guide

### Scenario 1 – Crop Recommendation
1. Go to the Home page and scroll to the **Prediction Form**.
2. Enter values for Nitrogen, Phosphorous, Potassium, Temperature, Humidity, pH, and Rainfall.
3. Click **Predict Best Crop**.
4. View the recommended crop, confidence score, suitable conditions, benefits, farming practices, and more.

### Scenario 2 – Suitability Check
1. Navigate to the **Suitability** page from the navbar.
2. Select a crop from the dropdown.
3. Enter your current environmental conditions.
4. Click **Check Suitability** to see the suitability score, advantages, issues, and suggestions.

### Scenario 3 – Research Dashboard
1. Navigate to the **Dashboard** page from the navbar.
2. Explore interactive charts for crop distribution, NPK analysis, environmental distributions, and the correlation heatmap.

---

## Screenshots

Add your screenshots in the `screenshots/` folder:

- `screenshots/home.png` – Home page with hero section
- `screenshots/predict.png` – Prediction form
- `screenshots/result.png` – Prediction result page
- `screenshots/suitability.png` – Suitability checker
- `screenshots/dashboard.png` – Research dashboard
- `screenshots/about.png` – About page
- `screenshots/darkmode.png` – Dark mode

---

## Future Enhancements

- Real-time weather API integration for automatic parameter detection
- Mobile app (React Native / Flutter)
- Multi-language support for regional farmers
- Soil image analysis using Deep Learning (CNN)
- IoT sensor integration for automated soil testing
- Crop rotation planning with multi-year recommendations
- Yield prediction with regression models
- Government scheme recommendations based on crop and region
- Voice-based input for accessibility
- Community forum for farmer knowledge sharing

---

## License

This project is created for the SmartBridge Virtual Internship program. Free to use for educational purposes.

---

## Acknowledgements

- **SmartBridge** – Virtual Internship Program
- Dataset inspired by the [Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset) on Kaggle
- Built with Flask, Scikit-learn, Bootstrap 5, and Chart.js
