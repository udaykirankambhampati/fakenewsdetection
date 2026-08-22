from flask import Flask, render_template, request, session
import joblib
import math
from utils.preprocessing import clean_text

app = Flask(__name__)
app.secret_key = "fake-news-detector-week4"

# Load trained model
model = joblib.load("model/fake_news_model.pkl")

# Load TF-IDF vectorizer
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    confidence = None

    history = session.get("history", [])

    if request.method == "POST":

        title = request.form["title"]
        content = request.form["content"]

        title = clean_text(title)
        content = clean_text(content)

        text = title + " " + content

        features = vectorizer.transform([text])

        # Prediction
        result = model.predict(features)[0]

        # Passive Aggressive Classifier does not have predict_proba()
        score = model.decision_function(features)[0]

        confidence = round(
            (1 / (1 + math.exp(-abs(score)))) * 100,
            2
        )

        if result == 0:
            prediction = "FAKE NEWS"
        else:
            prediction = "REAL NEWS"

        # Save prediction history
        history.insert(0, {
            "title": title,
            "prediction": prediction,
            "confidence": confidence
        })

        # Keep only latest 5 predictions
        history = history[:5]

        session["history"] = history

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        history=history
    )


@app.route("/clear-history")
def clear_history():

    session.pop("history", None)

    return render_template(
        "index.html",
        prediction=None,
        confidence=None,
        history=[]
    )


if __name__ == "__main__":
    app.run(debug=True)