🎬 Movie Recommendation Classification using Logistic Regression
📌 Project Overview

This project demonstrates how Machine Learning can be used to classify movies as recommended (Top Movie) or not recommended based on their popularity and audience engagement.
The model is built using Logistic Regression, a supervised learning algorithm.

The dataset used is the TMDB Movies Dataset, and the goal is to predict whether a movie should be recommended to users.

🧠 Machine Learning Concept Used

Type: Supervised Learning

Algorithm: Logistic Regression

Problem Type: Binary Classification

The model predicts whether a movie is:

✅ Top Movie (Recommended)

❌ Not a Top Movie

📂 Dataset Information

Dataset: tmdb_movies_dataset.csv

Columns Used:
Column Name	Description
vote_count	Number of votes received by the movie
popularity	Popularity score of the movie
rating	Average rating
top_movie	Target variable (created)
🎯 Target Variable Logic

A movie is considered a Top Movie if:

rating >= 6.5 AND vote_count >= 400

df['top_movie'] = ((df['rating'] >= 6.5) & (df['vote_count'] >= 400))


True → Recommended movie

False → Not recommended

🔢 Input Features

The model is trained using the following features:

X = ['vote_count', 'popularity']

Why these features?

Vote count shows audience engagement

Popularity reflects current interest and reach

🧪 Train-Test Split

The dataset is split into training and testing sets:

train_test_split(test_size=0.3)


70% Training data

30% Testing data

⚙️ Model Used
LogisticRegression()


Suitable for binary classification

Outputs probability-based predictions

📊 Model Evaluation
1️⃣ Accuracy Score
accuracy_score(y_test, y_predict)


Measures overall correctness of the model

2️⃣ Confusion Matrix

A confusion matrix is used to evaluate:

True Positives

True Negatives

False Positives

False Negatives

Visualization is done using Seaborn Heatmap:

seaborn.heatmap(cm, annot=True, fmt="d", cmap="Blues")


📌 This helps in understanding where the model is making correct or incorrect predictions.

📈 Visualization Output

X-axis: Predicted values

Y-axis: Actual values

Color: Darker = higher count

🛠️ Technologies & Libraries Used

Python

Pandas – Data manipulation

Scikit-learn – ML model & evaluation

Seaborn – Visualization

Matplotlib – Plot rendering

▶️ How to Run the Project
1️⃣ Install Dependencies
pip install pandas scikit-learn seaborn matplotlib

2️⃣ Run the Script
python movie_recommendation.py

✅ Final Outcome

Successfully classified movies into recommended and not recommended

Achieved a measurable accuracy score

Visualized performance using a confusion matrix

🚀 Future Improvements

Add more features (genre, runtime, budget)

Use advanced models like Random Forest or XGBoost

Convert this into a real recommendation system

👤 Author

Kaushal Raut
Machine Learning Intern
Passionate about Data Science & AI