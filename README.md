# Machine Learning Assignment 2: Mobile Price Classification

### a. Problem Statement
The objective is to classify mobile phones into four price tiers (0: Low Cost, 1: Medium Cost, 2: High Cost, 3: Very High Cost) using technical specifications. We implement, tune, and evaluate five benchmark algorithms: Logistic Regression, Decision Tree Classifier, K-Nearest Neighbors (KNN), Gaussian Naive Bayes, and Support Vector Machine (SVM). The models are serialized and integrated into a production Streamlit application hosted on Streamlit Community Cloud for interactive inference.

### b. Dataset Description
* **Dataset**: Mobile Price Classification 
* **Number of Instances**: 2,000 samples
* **Number of Input Features**: 20 technical features (`battery_power`, `ram`, `clock_speed`, etc.)
* **Target Variable**: `price_range` (Discrete categorical values: 0, 1, 2, 3)
* **Preprocessing**: Continuous variables are standardized using `StandardScaler` fitted on the training split to avoid data leakage; tree-based models use the unscaled features directly.

### c. Repository and Deployment Links
* **GitHub Repository Link**: [Insert your GitHub link here]
* **Live Streamlit App Link**: [Insert your Streamlit app link here]

### d. Models Used and Metric Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.9650 | 0.9987 | 0.9650 | 0.9650 | 0.9650 | 0.9534 |
| **Decision Tree** | 0.8325 | 0.9224 | 0.8343 | 0.8325 | 0.8327 | 0.7771 |
| **KNN** | 0.5625 | 0.7893 | 0.5753 | 0.5625 | 0.5674 | 0.4174 |
| **Naive Bayes** | 0.8100 | 0.9506 | 0.8113 | 0.8100 | 0.8105 | 0.7468 |
| **Support Vector Machine** | 0.9625 | 0.9985 | 0.9627 | 0.9625 | 0.9625 | 0.9501 |

### Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Performed exceptionally well. The linear boundary effectively captures the strong relationship between RAM, battery capacity, and price tier, achieving the highest accuracy of 96.50%. |
| **Decision Tree** | Captured direct rule thresholds but underperformed compared to linear boundaries, as orthogonal axis splits struggle to separate subtle continuous diagonal boundaries across 20 features. |
| **KNN** | Recorded the lowest performance metrics (56.25% accuracy). The high-dimensional feature space (20 features) likely introduced significant distance noise, causing neighborhood density to lose predictive value. |
| **Naive Bayes** | Showed moderate accuracy (81.00%) because it relies on the strong assumption of class-conditional feature independence, which is violated by correlated features like `px_height` / `px_width`. |
| **Support Vector Machine** | Achieved excellent results (96.25% accuracy, 0.9501 MCC), trailing Logistic Regression by only a fraction of a percent. The linear margin boundary separates the standardized multi-class hyperplanes with minimal misclassifications. |
| **Overall, Winner for your dataset?** | **Logistic Regression** narrowly takes the win for this dataset, leading with an Accuracy of 0.9650, AUC of 0.9987, and MCC of 0.9534. |
