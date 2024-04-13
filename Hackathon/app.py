from flask import Flask, render_template, request, redirect, url_for
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load and preprocess the dataset
data = pd.read_csv(r"IEEE Hackathon\Hackathon\updated_dataset.csv")

# Define label ranges and corresponding labels
def assign_label(score):
    if score >= 20:
        return 'depression'
    elif score >= 10:
        return 'anxiety'
    else:
        return 'stress'

# Add label column based on 'happiness_score'
data['label'] = data['happiness_score'].apply(assign_label)

# Initialize the model
model = DecisionTreeClassifier()

# Features (X) and labels (y)
X = data[['happiness_score']]
y = data['label']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train)

@app.route('/')
def login():
    return render_template('loginpage.html')

@app.route('/index', methods=['POST'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user inputs from the form
    user_answers = [int(request.form[f'answer_{i}']) for i in range(1, 11)]

    # Calculate happiness score by summing up user answers
    happiness_score = sum(user_answers)

    # Use the trained model to predict the mental status based on happiness score
    predicted_label = model.predict([[happiness_score]])

    return render_template('result.html', predicted_label=predicted_label)

@app.route('/submit', methods=['POST'])
def submit():
    # Assuming the form submission on index.html redirects to /submit
    return redirect(url_for('predict'))

if __name__ == '__main__':
    app.run(debug=True)
