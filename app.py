from flask import Flask, render_template, request
import pandas as pd
import seaborn as sns
import io
import urllib
import base64
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import pipeline

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def home():
    """Take a user selection from a drop down menu and pass the value to Plots.html"""
    games = ["No Man's Sky"]
    if request.method == "POST":
        # Get the user's selection from the drop down menu
        user_selection = request.form.get("game")
        # Read in the csv file
        data = pd.read_parquet("..\\..\\..\\data_pipe\\data\\No_Man's_Sky.parquet")
        # Filter the data based on the user's selection
        # df = data[data['game'] == user_selection]
        # make a seaborn plot of votes_up vs weighted_vote_score
        sns.scatterplot(x="votes_up", y="weighted_vote_score", data=data)
        png_image = io.BytesIO()
        plt.savefig(png_image, format="png")
        png_image.seek(0)
        plot = urllib.parse.quote(base64.b64encode(png_image.read()))
        # Pass the html table back to the Plots.html page
        return render_template(
            "Plots.html", plot=plot, game=user_selection, title="Game Results"
        )
    return render_template("home.html", games=games, title="Home")


@app.route("/Plots")
def Plots():
    return render_template("Plots.html", title="About Us")


@app.route("/Sentiment Analysis", methods=["GET", "POST"])
def sentiment():
    # get user input
    if request.method == "POST":
        classifier = pipeline("sentiment-analysis")
        user_input = request.form.get("user_input")
        print(f"user input is: {user_input}")
        # run sentiment analysis on user input
        results = classifier(user_input)
        sentiment = results[0]["label"]
        # pass sentiment back to html
        return render_template(
            "Sentiment_Analysis.html", sentiment=sentiment, title="Sentiment Analysis"
        )
    return render_template("Sentiment_Analysis.html", title="About Us")


if __name__ == "__main__":
    app.run(port = 8080, debug=True)
