import nltk
from flask import Flask, render_template, request
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from transformers import pipeline
import numpy as np
from nltk import trigrams
from nltk import bigrams
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from wordcloud import STOPWORDS, WordCloud
from plotly.subplots import make_subplots
from azure_sql_connect import get_dataframe


nltk.download("stopwords")
nltk.download("punkt")

def dropNAs(df):
    df_clean = df.dropna()
    return df_clean


def fit_kmeans(texts, num_clusters=3):
    """
    Fit a k-means clustering model on a list of texts.

    Parameters:
        texts (list): List of text data.
        num_clusters (int): Number of clusters for k-means (default is 3).

    Returns:
        tuple: A tuple containing the fitted k-means model and the TfidfVectorizer.
    """
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)

    kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(X)

    return kmeans, vectorizer


def generate_cluster_titles(kmeans, vectorizer, num_words_per_title=5):
    """
    This function creates computer generated titles based on the centroids.

    Input: kmeans model, vectorized reviews, num of words wanted in the title.

    Return: Computer generated cluster title strings
    """

    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]

    terms = vectorizer.get_feature_names_out()

    num_clusters = kmeans.n_clusters

    cluster_titles = []

    for i in range(num_clusters):
        cluster_terms = [terms[ind] for ind in order_centroids[i, :num_words_per_title]]

        title = " ".join(cluster_terms)

        cluster_titles.append(title)

    return cluster_titles


def generate_cluster_data(kmeans, texts):
    """
    This connects the cluster text to the kmeans model.

    Input:
        kmeans (object): Fitted k-means model.
        texts (list): List of text data (each element is a string representing a sentence).

    Return:
        dict: Cluster data with keys as cluster numbers and values as lists of sentences.
    """

    num_clusters = kmeans.n_clusters
    cluster_data = {}

    for i in range(num_clusters):
        # Get indices of sentences associated with the cluster
        cluster_indices = np.where(kmeans.labels_ == i)[0]

        # Ensure indices are within the valid range
        valid_indices = [idx for idx in cluster_indices if idx < len(texts)]

        # Get sentences for the valid indices
        cluster_sentences = [texts[idx] for idx in valid_indices]

        cluster_data[i] = cluster_sentences

    return cluster_data


def get_cluster_dataframe(cluster_data, cluster_titles, selected_cluster_index):
    """
    This puts cluster data into a dataframe with the computer generated title with a user selected cluster number.

    Return PD dataframe with cluster title and reviews for that cluster
    """
    # Ensure the selected_cluster_index is valid
    if selected_cluster_index < 0 or selected_cluster_index >= len(cluster_data):
        raise ValueError("Invalid selected_cluster_index")

    # Create a dataframe for the selected cluster with the corresponding title
    selected_cluster_title = cluster_titles[selected_cluster_index]
    selected_cluster_df = pd.DataFrame({"Review": cluster_data[selected_cluster_index]})
    selected_cluster_df[
        "Cluster Title"
    ] = selected_cluster_title  # Add the cluster title column

    return selected_cluster_df


def get_most_frequent_words(cluster_data):
    """
    This function gets keywords from each cluster, excluding the word "game".

    Input: Cluster reviews, tokenized by sentence, and max keywords.

    Return: gets top words from each cluster.
    """
    stop_words = set(stopwords.words("english"))
    most_frequent_words_per_cluster = {}

    for cluster_index, reviews in cluster_data.items():
        # Combine all reviews within the cluster into a single string
        combined_text = " ".join(reviews)

        # Tokenize the combined text into words
        words = word_tokenize(combined_text)

        # Remove stopwords and the word "game"
        filtered_words = [
            word.lower()
            for word in words
            if word.isalpha()
            and word.lower() not in stop_words
            and word.lower() != "game"
            and word.lower() != "get"
        ]

        # Calculate the frequency distribution of words
        freq_dist = FreqDist(filtered_words)

        # Get the most common words
        most_common_words = freq_dist.most_common(
            20
        )  # You can adjust the number as needed

        most_frequent_words_per_cluster[cluster_index] = most_common_words

    return most_frequent_words_per_cluster


def get_most_frequent_bigrams(cluster_data):
    """
    This function gets bigrams  from each cluster.

    Input: Cluster reviews, tokenized by sentence, and max keywords.

    Return: gets bigrams from each cluster.
    """
    stop_words = set(stopwords.words("english"))
    most_frequent_bigrams_per_cluster = {}

    for cluster_index, reviews in cluster_data.items():
        # Combine all reviews within the cluster into a single string
        combined_text = " ".join(reviews)

        # Tokenize the combined text into words
        words = word_tokenize(combined_text)

        # Remove stopwords
        filtered_words = [
            word.lower()
            for word in words
            if word.isalpha() and word.lower() not in stop_words
        ]

        # Get bigrams from the filtered words
        bigrams_list = list(bigrams(filtered_words))

        # Calculate the frequency distribution of bigrams
        freq_dist = FreqDist(bigrams_list)

        # Get the most common bigrams
        most_common_bigrams = freq_dist.most_common(
            10
        )  # You can adjust the number as needed

        most_frequent_bigrams_per_cluster[cluster_index] = most_common_bigrams

    return most_frequent_bigrams_per_cluster


def get_most_frequent_trigrams(cluster_data):
    """
    This function gets trigrams from each cluster.

    Input: Cluster reviews, tokenized by sentence, and max keywords.

    Return: gets trigrams from each cluster.
    """
    stop_words = set(stopwords.words("english"))
    most_frequent_trigrams_per_cluster = {}

    for cluster_index, reviews in cluster_data.items():
        # Combine all reviews within the cluster into a single string
        combined_text = " ".join(reviews)

        # Tokenize the combined text into words
        words = word_tokenize(combined_text)

        # Remove stopwords
        filtered_words = [
            word.lower()
            for word in words
            if word.isalpha() and word.lower() not in stop_words
        ]

        # Get trigrams from the filtered words
        trigrams_list = list(trigrams(filtered_words))

        # Calculate the frequency distribution of trigrams
        freq_dist = FreqDist(trigrams_list)

        # Get the most common trigrams
        most_common_trigrams = freq_dist.most_common(
            10
        )  # You can adjust the number as needed

        most_frequent_trigrams_per_cluster[cluster_index] = most_common_trigrams

    return most_frequent_trigrams_per_cluster


def get_most_representative_reviews(cluster_data):
    """
    This function gets the most representative review based off the cluster groupings

    Input: Cluster reviews, tokenized by sentence.

    Return: The most representative review of each cluster for a different title.
    """
    most_representative_reviews = {}

    for cluster_index, reviews in cluster_data.items():
        # Filter reviews that are 10 words or less
        short_reviews = [review for review in reviews if len(review.split()) <= 10]

        # If there are short reviews, select the first one; otherwise, choose the original reviews
        selected_review = short_reviews[0] if short_reviews else reviews[0]

        most_representative_reviews[cluster_index] = selected_review

    return most_representative_reviews


def filter_stop_words(review):
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(review)

    # Remove punctuation
    words = [word for word in words if word.isalnum()]

    filtered_review = [word for word in words if word.lower() not in stop_words]

    filtered_review = [review for review in filtered_review if review is not None]

    return " ".join(filtered_review)


def get_tokenized_texts(df):
    sample_texts = df["review"].tolist()
    filtered_reviews = [filter_stop_words(review) for review in sample_texts]
    return filtered_reviews


def generate_wordcloud_figures(single_word, representative_reviews):
    figures = []

    for cluster_id, word_freq_list in single_word.items():
        # Extract words and their frequencies from the list of tuples
        words, frequencies = zip(*word_freq_list)

        # Create a dictionary mapping words to frequencies
        word_freq_dict = dict(zip(words, frequencies))

        # Create a WordCloud object with custom styling
        wordcloud = WordCloud(
            width=400,
            height=300,
            background_color="white",
            colormap="twilight_shifted",
            max_words=200,
            contour_width=1,
            contour_color=None,
            stopwords=STOPWORDS,
        ).generate_from_frequencies(word_freq_dict)

        # Get the representative review for the current cluster
        representative_review = representative_reviews[cluster_id]

        # Convert WordCloud object to an image
        img_array = wordcloud.to_array()

        # Get the actual dimensions of the Word Cloud image
        img_width, img_height = wordcloud.width, wordcloud.height

        # Create a Plotly figure for each cluster
        fig = go.Figure()

        # Add the WordCloud image to the Plotly figure
        fig.add_trace(go.Image(z=img_array))

        title_text = (
            f"Word Cloud for Cluster {cluster_id}<br>Title: {representative_review}"
        )

        # Add title annotation
        fig.add_annotation(
            text=title_text,
            xref="paper",
            yref="paper",
            x=0.5,
            y=1.05,
            xanchor="center",
            yanchor="bottom",
            font=dict(size=16),
            showarrow=False,
        )

        # Hide axis
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)

        # Adjust the size of the figure based on the actual dimensions of the Word Cloud image
        fig.update_layout(
            width=img_width
            * 1.5,  # Adjust the width based on the Word Cloud image width
            height=img_height
            * 1.2,  # Adjust the height based on the Word Cloud image height
            margin=dict(t=60, b=10, l=0, r=0),  # Adjust top and bottom margins
        )

        figures.append(fig)

    return figures


def plot_trigram_word_phrases(trigram_word, representative_reviews):
    # Define custom colors related to gaming
    gaming_colors = ["#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD", "#8C564B"]

    # Create subplots with shared y-axes
    fig = make_subplots(
        rows=len(trigram_word),
        cols=1,
        subplot_titles=[
            f"<b>Top 3-word phrases for cluster {cluster_id}<br>Representative Review: {representative_reviews[cluster_id]}"
            for cluster_id in trigram_word.keys()
        ],
    )

    for i, (cluster_id, trigram_data) in enumerate(trigram_word.items(), start=1):
        trigrams, frequencies = zip(
            *[(f"{t[0][0]} {t[0][1]} {t[0][2]}", t[1]) for t in trigram_data]
        )

        # Convert frequencies to integers (if they're not already)
        frequencies = list(map(int, frequencies))

        # Create horizontal bar chart trace for each cluster
        trace = go.Bar(
            y=trigrams,
            x=frequencies,
            orientation="h",
            marker=dict(color=gaming_colors[cluster_id]),
        )

        # Add the trace to the subplot
        fig.add_trace(trace, row=i, col=1)

        # Update layout for each subplot
        fig.update_xaxes(
            title_text="Frequency",
            row=i,
            col=1,
            showgrid=True,
            gridwidth=0.5,
            gridcolor="lightgray",
        )
        fig.update_yaxes(
            title_text="Trigrams",
            row=i,
            col=1,
            showgrid=True,
            gridwidth=0.5,
            gridcolor="lightgray",
        )

    # Update overall layout
    fig.update_layout(
        height=300 * len(trigram_word),  # Adjust height based on the number of clusters
        showlegend=False,
        title_font_size=16,
    )

    return fig


def clustering_wordclouds_graphs(df):
    df = dropNAs(df)
    filtered_reviews = get_tokenized_texts(df)
    kmeans, vectorizer = fit_kmeans(filtered_reviews, num_clusters=3)
    data = generate_cluster_data(kmeans, filtered_reviews)
    single_word = get_most_frequent_words(data)
    rep_review = get_most_representative_reviews(data)
    # generate wordcould figures is dependent on get representative reviews

    return generate_wordcloud_figures(single_word, rep_review)


def clustering_trigrams_graphs(df):
    df = dropNAs(df)
    filtered_reviews = get_tokenized_texts(df)
    kmeans, vectorizer = fit_kmeans(filtered_reviews, num_clusters=3)
    data = generate_cluster_data(kmeans, filtered_reviews)
    trigrams = get_most_frequent_trigrams(data)
    rep_reviews = get_most_representative_reviews(data)

    # Generate trigram word phrases plot within this function
    fig = plot_trigram_word_phrases(trigrams, rep_reviews)

    # Return the generated figure
    return fig


app = Flask(__name__)


def create_plot(df, grouping_selection, start_date, end_date):
    fig = go.Figure()
    # convert 'timestamp_created' to integer

    df["date_time"] = df["timestamp_created"]

    if grouping_selection == "Month":
        df["year_month"] = df["date_time"].dt.to_period("M")
        grouped_counts = df.groupby("year_month")["voted_up"].value_counts().unstack()
        grouped_counts = grouped_counts.reset_index()
        grouped_counts["year_month"] = grouped_counts["year_month"].dt.strftime("%b %Y")

        positive_counts = grouped_counts[True]
        negative_counts = -grouped_counts[False]

        fig.add_trace(
            go.Bar(
                x=grouped_counts["year_month"],
                y=positive_counts,
                name="Positive",
                marker_color="turquoise",
            )
        )

        fig.add_trace(
            go.Bar(
                x=grouped_counts["year_month"],
                y=negative_counts,
                name="Negative",
                marker_color="salmon",
            )
        )

        fig.update_layout(
            title="Positive and Negative Counts by Month",
            xaxis_title="Month",
            yaxis_title="Count",
            barmode="relative",
        )

    elif grouping_selection == "Day":
        grouped_counts = df.groupby("date_time")["voted_up"].value_counts().unstack()
        grouped_counts = grouped_counts.reset_index()

        positive_counts = grouped_counts[True]
        negative_counts = -grouped_counts[False]

        fig.add_trace(
            go.Bar(
                x=grouped_counts["date_time"],
                y=positive_counts,
                name="Positive",
                marker_color="turquoise",
            )
        )

        fig.add_trace(
            go.Bar(
                x=grouped_counts["date_time"],
                y=negative_counts,
                name="Negative",
                marker_color="salmon",
            )
        )

        fig.update_layout(
            title="Positive and Negative Counts by Day",
            xaxis_title="Day",
            yaxis_title="Count",
            barmode="relative",
        )
    return fig


@app.route("/", methods=["GET", "POST"])
def home():
    """Take a user selection from a drop down menu and pass the value to Plots.html"""
    games = [
        "No Man's Sky",
        "Phasmaphobia",
        "Rust",
        "Dead By Daylight",
        "Stardew Valley",
        "Fallout 4",
        "Sea of Thieves",
    ]
    grouping = ["day", "month"]
    if request.method == "POST":
        # Get the user's selection from the drop down menu
        user_selection = request.form.get("game")
        user_grouping = request.form.get("grouping_selection")
        user_start_date = request.form.get("start_date")
        user_end_date = request.form.get("end_date")

        # The following is the query that will be ran to get the dataframe
        final_query = f"""
        SELECT * FROM default.final_steam_table 
        WHERE game_name = "{user_selection}" 
        AND timestamp_created > "{user_start_date}" 
        AND timestamp_created < "{user_end_date}"
        """

        # Read in the csv file
        #data = pd.read_parquet("../data/No_Man's_Sky_clean.parquet")

        data = get_dataframe(final_query)

        # Filter the data based on the user's selection
        # df = data[data['game'] == user_selection]
        # make a seaborn plot of votes_up vs weighted_vote_score
        plot = create_plot(data, user_grouping, user_start_date, user_end_date)
        plots = clustering_wordclouds_graphs(data)
        plot1 = clustering_trigrams_graphs(data)
        plot2 = plots[0]
        plot3 = plots[1]
        plot4 = plots[2]
        print(plot4)
        plot_html = pio.to_html(plot, full_html=False, include_plotlyjs="cdn")
        plot_html1 = pio.to_html(plot1, full_html=False, include_plotlyjs="cdn")
        plot_html2 = pio.to_html(plot2, full_html=False, include_plotlyjs="cdn")
        plot_html3 = pio.to_html(plot3, full_html=False, include_plotlyjs="cdn")
        plot_html4 = pio.to_html(plot4, full_html=False, include_plotlyjs="cdn")
        return render_template(
            "Plots.html",
            plot=plot_html,
            plot1=plot_html1,
            plot2=plot_html2,
            plot3=plot_html3,
            plot4=plot_html4,
            grouping=grouping,
            game=user_selection,
            title="Game Results",
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
    app.run(debug=True, host="0.0.0.0", port=5000)
