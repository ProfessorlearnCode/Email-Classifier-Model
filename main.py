import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    filtered_words = [word for word in tokens if word.isalnum() and word not in stopwords.words('english')]
    return ' '.join(filtered_words)

# Sample input
sample_text = "This is an example email!"
processed_text = preprocess_text(sample_text)
print("Processed Sample Text:", processed_text)

# Load dataset
data = pd.read_csv('spam.csv', encoding='latin-1')  # Specify encoding if necessary
data = data[['v1', 'v2']]  # Assuming the columns are named 'v1' for Category and 'v2' for Message
data.columns = ['Category', 'Message']
data['Spam'] = data['Category'].apply(lambda x: 1 if x == 'spam' else 0)

# Preprocess the message text
data['Message'] = data['Message'].apply(preprocess_text)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(data['Message'], data['Spam'], test_size=0.25, random_state=42)

# Create a pipeline and train the model
clf = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('nb', MultinomialNB())
])
clf.fit(X_train, y_train)

# Sample emails for prediction
emails = [
    'Sounds great! Are you home now?',
    'Will u meet ur dream partner soon? Is ur career off 2 a flyng start? 2 find out free, txt HORO followed by ur star sign, e. g. HORO ARIES'
]
predictions = clf.predict(emails)

# Print the predictions
print("Predictions:", predictions)

# Evaluate the model on the test data
accuracy = clf.score(X_test, y_test)
print("Accuracy on test data:", accuracy)
