# Streamlit app

import pickle
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score
from sklearn.model_selection import train_test_split

model = load_model('model.pkl')  # Load your MultinomialNB model
vectorizer = load_vectorizer('vectorizer.pkl')  # Load your TfidfVectorizer

def main():
    st.title("SMS Spam Classifier")

    # Input text box
    input_text = st.text_area("Enter your text here:", "")

    # Submit button
    if st.button("Submit"):
        if input_text:
            # Preprocess the input text
            processed_text = transform_text(input_text)

            # Vectorize the processed text
            vectorized_text = vectorizer.transform([processed_text])

            # Make prediction using the model
            prediction = model.predict(vectorized_text)

            # Display the result
            result = "Spam" if prediction[0] == 1 else "Not Spam"
            st.success(f"Prediction: {result}")

# Run the Streamlit app
if __name__ == "__main__":
    main()
