**Goal:** trying to predict whether a given ramen will be rated as Top 10 or not. We will be using a multi-input TensorFlow neural network to make our predictions.


**Data:** The dataset used in this project is the Ramen Rating dataset from Kaggle. The dataset contains the following columns: Brand, Variety, Style, Country, Stars, Top Ten. The dataset contains 2580 rows and 7 columns. The dataset is available at https://www.kaggle.com/residentmario/ramen-ratings.


**Methodology:** The dataset is preprocessed and split into training and testing datasets. The neural network is built using TensorFlow and Keras. The neural network has two inputs: one for the brand and one for the variety of the ramen. The neural network has two hidden layers with 64 and 32 neurons, respectively. The output layer has one neuron with a sigmoid activation function. The model is trained using the training dataset and evaluated using the testing dataset. The model is then used to make predictions on new data.

---

**Results:** The model is able to predict whether a given ramen will be rated as Top 10 or not with an accuracy of 91%.


