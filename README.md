# Weather-Driven Electricity Load Clustering + Load Category Prediction (Ireland)

## Overview
This project predicts Ireland’s electricity demand patterns (2015–2020) by combining weather analytics and machine learning. It first discovers “types of days” from weather data using clustering, assigns those cluster labels to national data, and then classifies electricity load into Low / Medium / High using tuned tree-based models.

## Data
- Electricity consumption at 30-minute intervals (converted to hourly level for modelling)
- Weather data with multiple station and national weather variables (filtered to 2015–2020)
- Weather variables used include rain, temperature, humidity, pressure, wind, sunshine, visibility, and cloud metrics

## Tools & Libraries
- pandas, numpy (data processing)
- matplotlib, seaborn (visualisation)
- scipy (Ward linkage clustering)
- scikit-learn:
  - StandardScaler
  - KMeans (clustering)
  - KNN (cluster assignment)
  - Decision Tree / Random Forest / Gradient Boosting (classification)
  - GridSearchCV (hyperparameter tuning)
  - silhouette_score + classification_report + confusion_matrix (evaluation)
- joblib (saving models)

## Methodology

### 1) Preprocessing and Feature Engineering
- Merge and clean weather + station datasets
- Convert load data to hourly level
- Remove extreme outliers in weather variables
- Create time features: Hour, DayOfWeek, Month, Year

### 2) Weather Pattern Clustering
- Ward linkage (hierarchical clustering) to produce cluster labels
- K-Means tested with silhouette scoring to select k (k=3 performed best in the notebook)
- Clusters represent different weather-day patterns

### 3) Assign Clusters Nationally
- KNN classifier trained on clustered data
- Used to label national weather records with the learned cluster IDs

### 4) Load Category Prediction
- Convert continuous load into Low/Medium/High using quantiles (qcut, q=3)
- Predict load category using:
  - Decision Tree (tuned with GridSearchCV)
  - Random Forest (tuned with GridSearchCV)
  - Gradient Boosting (tuned with GridSearchCV)

## Results (Highlights)
- Decision Tree achieved solid performance (~0.88 accuracy).
- Random Forest and Gradient Boosting performed best (~0.91 accuracy).
- Evaluation includes precision/recall/f1 per class and confusion matrices.
- Trained models are saved using joblib for reuse.

## Outcome
A complete pipeline combining clustering + supervised learning to produce interpretable, weather-informed electricity demand classification for Ireland, with model tuning and saved artefacts for future deployment.
