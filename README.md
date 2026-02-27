# ING Hubs Datathon - Team Kolay

This project was developed for the ING Hubs Datathon by **Team Kolay**. Our main goal is finding potential customer loss (churn) in the banking sector.

## Team Members
* **Batuhan Yılmaz** - Yildiz Technical University
* **Ahmet Uzoğlu** - Istanbul Technical University

## Project Flow and Methodology
The project is built as an end-to-end data processing and modeling pipeline. Running the `src/main.py` script executes the following steps in order:

1. **Data Loading and Preprocessing:** Customer and historical data are loaded, and initial preprocessing steps are applied.
2. **Feature Engineering:** High-impact features are created to improve model performance.
3. **Time-based Data Splitting:** The dataset is split into training and validation sets based on time.
4. **Hyperparameter Optimization:** The best parameters for the model are automatically found using the `Optuna` library.
5. **Model Training and Prediction:** The final model is trained, evaluated on the validation set, saved, and final predictions are generated.

## Key Technologies Used
* **Modeling:** LightGBM, XGBoost, CatBoost, Scikit-learn
* **Data Processing & Analysis:** Pandas, NumPy, SciPy
* **Optimization:** Optuna
* **Visualization:** Matplotlib, Seaborn, Plotly

## Installation and Usage

To run this code on your own workspace, please follow the steps below:

### 1. Install Required Libraries
Open a terminal in the project directory and run the following command to install all necessary dependencies:

pip install -r requirements.txt

### 2. Prepare the Data Folder
If you want to run this code on your workspace, you must create a data/raw folder in the root directory and upload your raw data files there.

### 3. Run the Pipeline
After loading the data, execute the main script to start the entire machine learning process:

python src/main.py
