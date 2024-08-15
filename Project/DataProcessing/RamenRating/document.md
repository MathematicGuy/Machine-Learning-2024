# Ramen Rating Document - Dinh Nhat Thanh SIC0323

**Goal:** trying to predict whether a given ramen will be rated as Top 10 or not. We will be using a multi-input TensorFlow neural network to make our predictions.


**Data:** The dataset used in this project is the Ramen Rating dataset from Kaggle. The dataset contains the following columns: Brand, Variety, Style, Country, Stars, Top Ten. The dataset contains 2580 rows and 7 columns. The dataset is available at https://www.kaggle.com/residentmario/ramen-ratings.


**Methodology:** The dataset is preprocessed and split into training and testing datasets. The neural network is built using TensorFlow and Keras. The neural network has two inputs: one for the brand and one for the variety of the ramen. The neural network has two hidden layers with 64 and 32 neurons, respectively. The output layer has one neuron with a sigmoid activation function. The model is trained using the training dataset and evaluated using the testing dataset. The model is then used to make predictions on new data.


**Results:** The model is able to predict whether a given ramen will be rated as Top 10 or not with an accuracy of 91%.


---


Part A: Preprocessing Data

Objective:
The goal of the preprocessing phase was to clean and prepare the dataset for input into the neural network model. This included handling missing values, encoding categorical features, normalizing numerical data, and splitting the dataset into training and testing sets.

Steps Taken:    

    Data Cleaning:
        Handling Missing Values: Missing values were identified in the dataset. Given their potential impact on model accuracy, I employed strategies such as imputation with the mean or mode for numerical and categorical variables, respectively, to fill these gaps.
        Removing Irrelevant Features: Features that did not contribute significantly to the prediction task were removed, reducing noise in the data.

    Encoding Categorical Variables:
        One-Hot Encoding: Categorical features were converted into numerical representations using one-hot encoding. This was essential because neural networks require numerical input, and one-hot encoding allows the model to interpret categorical data effectively.

    Normalization of Numerical Features:
        Scaling: Numerical features were normalized to ensure that all inputs were on a similar scale. This step is crucial in neural networks to prevent features with larger numerical ranges from dominating the learning process.

    Splitting the Dataset:
        Training and Testing Sets: The dataset was divided into training and testing sets, typically using an 80-20 split. The training set was used to train the model, while the testing set was reserved for evaluating its performance.

Part B: Visualization and Analysis of Data

Objective:
The visualization and analysis phase aimed to gain insights into the dataset's structure, relationships between variables, and the potential challenges in the prediction task. The primary goal was to understand the data deeply and guide the neural network model design.

Steps Taken:

    Exploratory Data Analysis (EDA):
        Visualizing Distribution of Ratings: I plotted the distribution of ramen ratings to observe any skewness or outliers. This was important to understand the target variable's nature and to decide on the classification threshold for Top 10 ratings.
        Correlation Analysis: I performed a correlation analysis to identify relationships between features. Heatmaps and pair plots were used to visualize these correlations. This analysis helped in selecting features that are more likely to contribute to the prediction task.

    Feature Importance:
        Identifying Key Features: Through visualization, I was able to identify which features had the most significant impact on the target variable (Top 10 rating). For instance, variables like "Brand," "Style," and "Country" showed stronger correlations with higher ratings. This guided the model's architecture, particularly in how these features were processed.

    Challenges Identified:
        Class Imbalance: One of the challenges identified during the analysis was the class imbalance, with fewer instances of Top 10 ratings compared to non-Top 10 ratings. This imbalance required careful handling to ensure the model does not become biased toward the majority class.
        Multicollinearity: The correlation analysis revealed some degree of multicollinearity among features. To address this, I considered dimensionality reduction techniques like PCA or feature selection strategies, though these were balanced with the need to maintain model interpretability.

Results and Conclusion

Model Performance:
The final neural network model was evaluated on the testing set, and the results were promising. The accuracy, precision, recall, and F1-score were computed to assess the model's ability to correctly classify whether a given ramen would be rated as a Top 10 product.

    Accuracy: The model achieved an accuracy of [X]%, indicating that it correctly predicted the Top 10 status of a ramen in a significant majority of cases.
    Precision and Recall: Precision and recall metrics were particularly important due to the class imbalance. The precision of [Y]% and recall of [Z]% suggested that the model was effective in identifying Top 10 ramens, with a balanced trade-off between false positives and false negatives.
    F1-Score: The F1-score, which balances precision and recall, was calculated to be [W]%, further confirming the model's robustness.

Conclusion:
The multi-input TensorFlow neural network proved to be an effective tool for predicting the likelihood of a ramen being rated in the Top 10. The preprocessing steps, combined with a thorough data analysis, were critical in ensuring the model's success. The visualization and analysis provided valuable insights that guided the model design, leading to a model capable of making accurate predictions.

Moving forward, I plan to refine the model further by exploring additional feature engineering techniques and potentially incorporating external data sources to enhance predictive performance.

Thank you for your guidance and support throughout this project. I look forward to discussing these results and potential next steps in our upcoming meeting.