import pickle
from preprocessing import preprocess
from sentiment import get_sentiment

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

def predict_news(text):
    processed = preprocess(text)
    vectorized = vectorizer.transform([processed])

    prediction = model.predict(vectorized)[0]
    sentiment = get_sentiment(text)

    return prediction, sentiment


# ===== USER INPUT =====
print("Fake News Detection + Sentiment Analysis")
print("----------------------------------------")

news = input("Enter news text:\n")

label, sentiment = predict_news(news)

print("\nResult:")
print("Fake/Real:", label)
print("Sentiment:", sentiment)