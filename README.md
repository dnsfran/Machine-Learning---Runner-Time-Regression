# Race Time Prediction — Aalto University Machine Learning Project

This project was developed as part of a **Machine Learning course at Aalto University**.  
The objective is to **predict the time it takes for a runner to finish a race**, based on the athlete’s characteristics and event information.

---

##  Project Overview

The dataset contains records of ultra-marathon races, over 7M race records, it has been provided by Kaggle "The big dataset of ultra-marathon running".

### Preprocessing Steps

- **Dataset loading**  
   Load the original CSV file containing race and athlete information.

- **Filtering races**  
   Keep only races that took place in **2020**.

- **Selecting fixed-distance races**  
   Remove races defined by time limits (e.g. “6-hour race”) and keep only those with a measurable distance (in km or miles).

- **Distance normalization**  
   Convert all event distances to **kilometers**, ensuring consistent measurement across events.

- **Outlier removal**  
   Within each event, remove the **top 5% fastest runners** based on average speed to reduce bias from extreme performances.

---


##  Goal
The cleaned data will be used to train **machine learning models** (e.g., regression algorithms) that predict an athlete’s **race finish time** based on: 
- Race characteristic 
- Athlete’s characteristics

---

## Technologies 
For this project, a polynomial regression model will be used. It provides a simple and interpretable baseline for understanding how age, gender, and distance relate to performance. It is also more appropriate than a linear regression, as the mean time of 200 km will not be 4 times the time of 50 km.


As a second method, we implemented a Random Forest Regressor. Random Forest is an ensemble method that builds multiple decision trees on random subsets of the data and features, and averages their outputs to produce the final prediction. This makes the model more robust than a single decision tree, as it reduces variance and improves generalization.
The motivation for using this method is that Random Forests can capture non-linear relationships between age, gender, and distance without requiring explicit feature transformations. They are also less sensitive to outliers and provide an interpretable measure of feature importance, which can indicate the relative contribution of each variable to performance prediction.


The dataset is divided into three parts: 70\% for training, 15\% for validation, and 15\% for testing. This split ensures a good balance between the need for sufficient training data, enough validation data to guide the selection of hyperparameters such as the polynomial degree, and a final independent test set to assess the generalization performance of the model. With more than 190,000 datapoints available, the training set remains large enough for stable learning, while the validation and test sets are representative of the underlying data distribution.



## Output

This project explored two machine learning approaches---polynomial regression and Random Forest regression---for predicting ultramarathon athletes’ average speed from age, gender, and race distance. The results indicate that polynomial regression with degree 4 provides the best generalization performance, achieving lower validation and test errors compared to Random Forest models.

The findings highlight two main points:

- Even relatively simple models such as polynomial regression can perform competitively when the feature space is limited but relationships are non-linear.
- More complex ensemble methods like Random Forests may not yield better accuracy unless additional informative features are incorporated.

There is still room for improvement. Future work could include experimenting with additional predictors (e.g., training history, altitude profile of the race, or weather conditions), testing other non-linear models such as gradient boosting, and applying regularization techniques to polynomial regression to further control overfitting. Collecting a more diverse dataset, including different years and event formats, may also enhance generalization.

In summary, the chosen polynomial regression model demonstrates that athlete performance in ultramarathons can be predicted with reasonable accuracy using only a few demographic and race-related features, while also leaving opportunities for richer modeling in future research.

