import streamlit as st
import pickle

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="🎬 AI Sentiment Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Load Model
# -----------------------------
with open("model/sentiment_model.pkl", "rb") as f:
    sentiment_model = pickle.load(f)

with open("model/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)


# -----------------------------
# Prediction Function
# -----------------------------
def predict_sentiment(review):
    review_vector = vectorizer.transform([review])

    prediction = sentiment_model.predict(review_vector)[0]

    probabilities = sentiment_model.predict_proba(review_vector)[0]

    confidence = max(probabilities)

    positive_prob = probabilities[1]
    negative_prob = probabilities[0]

    if prediction == 1:
        sentiment = "Positive 😊"
    else:
        sentiment = "Negative 😞"

    return sentiment, confidence, positive_prob, negative_prob


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:

    st.title("🤖 AI Sentiment")

    st.divider()

    st.success("Machine Learning Project")

    st.write("### Technologies")

    st.write("✅ Python")
    st.write("✅ Streamlit")
    st.write("✅ Scikit-Learn")
    st.write("✅ TF-IDF")
    st.write("✅ Logistic Regression")

    st.divider()

    st.metric("Model", "Logistic Regression")
    st.metric("NLP", "TF-IDF")

    st.divider()

    st.info(
        "Enter any movie review and let AI predict the sentiment."
    )


# -----------------------------
# Header
# -----------------------------
st.title("🎬 AI Movie Review Sentiment Analyzer")

st.caption(
    "Analyze movie reviews using Natural Language Processing and Machine Learning"
)

st.divider()


# -----------------------------
# Example Reviews
# -----------------------------
st.subheader("📌 Try an Example")

col1, col2, col3 = st.columns(3)

example = ""

with col1:
    if st.button("😊 Positive"):
        example = "I absolutely loved this movie. The acting was amazing."

with col2:
    if st.button("😞 Negative"):
        example = "Worst movie ever. It was boring and a waste of time."

with col3:
    if st.button("🤔 Mixed"):
        example = "The acting was good but the story was very slow."


# -----------------------------
# User Input
# -----------------------------
review = st.text_area(
    "✍️ Enter Movie Review",
    value=example,
    height=200,
    placeholder="Type your movie review here..."
)

st.divider()


# -----------------------------
# Predict Button
# -----------------------------
if st.button("🚀 Analyze Sentiment", use_container_width=True):

    if review.strip() == "":

        st.warning("Please enter a movie review.")

    else:

        sentiment, confidence, positive_prob, negative_prob = predict_sentiment(review)

        st.divider()

        if "Positive" in sentiment:

            st.success("🎉 Positive Review Detected")

            st.balloons()

        else:

            st.error("😞 Negative Review Detected")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Prediction",
                sentiment
            )

        with col2:
            st.metric(
                "Confidence",
                f"{confidence*100:.2f}%"
            )

        st.progress(float(confidence))

        st.subheader("📊 Prediction Probability")

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.metric(
                "😊 Positive",
                f"{positive_prob*100:.2f}%"
            )
            st.progress(float(positive_prob))

        with chart_col2:
            st.metric(
                "😞 Negative",
                f"{negative_prob*100:.2f}%"
            )
            st.progress(float(negative_prob))

        st.divider()

        with st.expander("📖 Review Entered"):

            st.write(review)

        with st.expander("ℹ️ About the Prediction"):

            st.write(
                """
This prediction is generated using:

- Logistic Regression
- TF-IDF Vectorizer
- Natural Language Processing (NLP)

The confidence score indicates how certain the model is about its prediction.
"""
            )

st.divider()

st.subheader("⭐ Project Features")

feature1, feature2, feature3 = st.columns(3)

with feature1:
    st.info("🤖 Machine Learning")

with feature2:
    st.info("📝 NLP")

with feature3:
    st.info("⚡ Real-Time Prediction")

st.divider()

st.caption("Made with ❤️ using Python, Scikit-Learn and Streamlit")