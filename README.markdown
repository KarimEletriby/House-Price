# House Price Prediction

## Project Overview
This project predicts house prices using the California Housing dataset with Machine Learning. It loads the data, explores it, preprocesses features, trains an XGBoost Regressor model, and evaluates performance using R² score and Mean Absolute Error (MAE).

## Dataset
The dataset is the **California Housing** dataset from scikit-learn, derived from the 1990 U.S. census. It includes:
- `MedInc`: Median income in block group
- `HouseAge`: Median house age in block group
- `AveRooms`: Average number of rooms per household
- `AveBedrms`: Average number of bedrooms per household
- `Population`: Block group population
- `AveOccup`: Average number of household members
- `Latitude`: Block group latitude
- `Longitude`: Block group longitude
- Target: `MedHouseVal` (Median house value in hundreds of thousands of dollars)

The dataset is fetched directly via `sklearn.datasets.fetch_california_housing()`.

## Project Structure
- `Project 4 (House Price).ipynb`: The main Jupyter notebook with data loading, exploration, preprocessing, model training, and evaluation.
- `README.md`: This file.
- (Optional) `requirements.txt`: List of dependencies.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/house-price-prediction.git
   cd house-price-prediction
   ```

2. Set up a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   Or manually:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost
   ```

## Data Preprocessing
- Load the dataset and create a DataFrame.
- Explore data: Head, shape, info, describe, check for null values, and correlation heatmap.
- No missing values or categorical features to encode.
- Split data into features (X) and target (y), then into train/test sets (80/20 split).

## Model and Evaluation
- Train an XGBoost Regressor model on the training data.
- Evaluate on the test set using:
  - R² Score: Measures the proportion of variance explained by the model.
  - Mean Absolute Error (MAE): Average absolute difference between predicted and actual values.

## Usage
1. Open and run the Jupyter notebook:
   ```bash
   jupyter notebook "Project 4 (House Price).ipynb"
   ```
2. The notebook will:
   - Fetch and explore the dataset.
   - Preprocess and split the data.
   - Train the XGBoost model.
   - Make predictions and compute evaluation metrics (R² ~0.81, MAE ~0.306).

## Results
- Model: XGBoost Regressor
- Test R² Score: ~0.81 (indicating good fit)
- Test MAE: ~0.306 (in units of $100,000)

## Future Improvements
- Feature engineering (e.g., create new features from latitude/longitude or handle outliers).
- Hyperparameter tuning for XGBoost using GridSearchCV.
- Try other regression models (e.g., Random Forest, Linear Regression) for comparison.
- Cross-validation for more robust evaluation.
- Visualize predictions vs. actual values with scatter plots.

## Contributing
Contributions are welcome! Fork the repository and submit a pull request.

## License
This project is licensed under the MIT License.

## Contact
For questions, open an issue on GitHub.