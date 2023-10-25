from collections import namedtuple
import altair as alt
import math
import streamlit as st

"""
# Welcome to Streamlit!

Edit `/streamlit_app.py` to customize this app to your heart's desire :heart:

If you have any questions, checkout our [documentation](https://docs.streamlit.io) and [community
forums](https://discuss.streamlit.io).

In the meantime, below is an example of what you can do with just a few lines of code:
"""

# Import necessary libraries
import numpy as np  # For linear algebra operations
import pandas as pd  # For data processing and CSV file I/O
import tensorflow as tf  # For creating the machine learning model
import matplotlib.pyplot as plt  # For data visualization
import seaborn as sns  # For data visualization
import re, string  # For data cleaning
from sklearn.preprocessing import LabelEncoder  # For data preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer  # For text vectorization
from sklearn.metrics import classification_report, accuracy_score, roc_curve, confusion_matrix  # For model evaluation
from sklearn.model_selection import train_test_split  # For splitting the data into training and testing sets

# Function to load the dataset from a URL
def load_dataset(url):
    df = pd.read_csv(url)
    return df

url = 'https://raw.githubusercontent.com/29Aegis62/suicide_dataset/main/Suicide_Ideation_Dataset(Twitter-based).csv?token=GHSAT0AAAAAACJIFASAXRG76KWL6XS6TPFSZJYX3TA'
df = load_dataset(url)

df.shape

df.head()

# Check for and display the number of missing values in the dataset
df.isnull().sum()

# Display the distribution of the 'Suicide' column
df['Suicide'].value_counts()

#Create a pie chart to visualize the target distribution
plt.figure(figsize=(9, 5))
palette = ['#ffb4a2', '#b5838d']
df["Suicide"].value_counts().plot(kind='pie', autopct='%1.0f%%', colors=palette)
plt.ylabel("")
plt.title("Target Distribution")
plt.show()

df = df.dropna() # As number of empty rows is quite small we can simply remove them
# Check for and display the number of missing values again
df.isnull().sum()

def cleaner(raw):

    # convert to lower case
    processed_sent = str(raw).lower()

    # remove user mentions & urls
    processed_sent = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", processed_sent)

    # remove special chars
    processed_sent = re.sub(r'\W', ' ', str(processed_sent))

    # remove single characters from the start
    processed_sent = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_sent)

    # remove hashtags but keep the text
    processed_sent = processed_sent.replace("#", "").replace("_", " ")

    # remove digits
    processed_sent = re.sub(r'\d+', '', str(processed_sent))

    # remove non alphanumeric chars
    processed_sent = ' '.join(e for e in processed_sent.split(' ') if e.isalnum())
    processed_sent = re.sub(r'[^A-Za-z0-9]+', ' ', str(processed_sent))

    # remove single chars
    processed_sent = re.sub(r'\s+[a-zA-Z]\s+', ' ', str(processed_sent))

    # remove punctuations
    punct = list(string.punctuation)
    special_punct=['©', '^', '®',' ','¾', '¡','!']
    punct.extend(special_punct)
    for p in punct:
        if p in processed_sent:
            processed_sent = processed_sent.replace(p, ' ')

    return processed_sent.lower()

# apply the function
df['Tweet'] = df['Tweet'].apply(lambda x: cleaner(x))

# view
df.sample(5)

# Encode the 'Suicide' column using LabelEncoder
lab_e = LabelEncoder()
df['Suicide'] = lab_e.fit_transform(df['Suicide'])
df.head()

# Vectorize the text data using TF-IDF vectorization with a maximum of 2000 features
vector = TfidfVectorizer(max_features=2000)
X = vector.fit_transform(df['Tweet'])
Y = df.Suicide
X = X.toarray()

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=26)

# Create a sequential neural network model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model on the training data
model.fit(X_train, Y_train, epochs=5, batch_size=32)


# Make predictions on the test data
Y_pred = model.predict(X_test)
Y_pred = np.round(Y_pred)

# Calculate and display the accuracy score
score = accuracy_score(Y_pred, Y_test)
print(f"Test Score: {score:.2f}")

# Define a function to plot the ROC curve
def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()


# Plot the ROC curve
fpr, tpr, thresholds = roc_curve(Y_test, Y_pred)
plot_roc_curve(fpr, tpr)

# Create a heatmap of the confusion matrix
sns.heatmap(confusion_matrix(Y_test, Y_pred), annot=True)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
