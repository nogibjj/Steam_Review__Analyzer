# Steam Review Analytics Dashboard

## A dashboard created ahering to DevOps principles, created to provide various insights based off of reviews left on gaming market place Steam, and some of its biggest games

### Contributions by: Kian Bagherlee, Yabei Zeng, Katelyn Hucker, John Coogan, and Suim Park

The following in a comprehensive report that will explain the full Data Engineering project, including its functionality, its limitations, and the tools used. 

### Quick Overview

This analytics Dashboard was created to be able to derive insights from the reviews left under certain large games in the Steam marketplace. Steam is an open-source gaming marketplace that provides the feature for users to leave positive and negative reviews under the various games. These reviews are presented as JSON files when utilizing Steam's API, which will provide the review, and other accompanying data for it.

This dashboard is a Python written interactable microservice, which utilizes an ETL pipeline in Azure Databricks. The microservice is contained in DockerHub, easily reproducible with the provided DockerImage, and hosted on Azure Web App to a public endpoint. It presents the information as an interactable website, created using Flask. It accepts user input, and will then process various types of graphs computed with the parameters requested from the user (these parameters are used in a SQL query to derive the proper information).

### The Games Analyzed

The games chosen were due to both their size and their diversity. The goal was to get data on different game genres, as different genres vary in both what people look for paired with what criticisms are valid, as well as different sizes. Larger titles, from well established AAA companies, tend to have certain stigmas around them that spoil the reviews, compared to indie games. Lastly, the games were chosen based on their review distribution. The goal was to choose a collection of games that are applauded for being well made, and games that were degraded for a myriad of reasons. In the end, the following games were selected in the end.

* Dead by Daylight
* Fallout 4
* Stardew Valley
* Rust
* No Man's Sky
* Sea of Thieves
* Phasmaphobia

### The ETL Pipeline

The data for this entire project is first derived **SOMEONE WHO DID THE SCRAPING, ENTER THIS HERE**

From here, the team created an ETL pipeline in Azure Databricks shown below. Set to run every week, it will **SOMEONE EXPLAIN THE PREPPING IT DOES**.

**_ENTER PHOTO OF AZURE DATABRICKS PIPELINE_**

The result of this ETL pipeline will create our Delta Table, called '''final_steam_table''', which will house every single review for every single game. This will also hold the variety of columns with information that the team used to derive analytical insights, as well as what game the review is actually attached too, which is necessary for identification.

### Analytics Derived

The following below is a brief description of each graph shown, and why.

- **Positive vs Negative Reviews**
- **Playtime at the time of Review Written**
- **Review Wordmap**

### Dashboard Display

The dashboard is fully displayed utilizing the Python package ```Flask```, which is an easy way to combine written Python code and HTML code. The HTML code is necessary to have an attractive interface, with simple UI features, to facilitate the information for thet user. The HTML code was written to have a display page that requests a selection of a game, a date range of interest, and a month/day of interest. 

From here, the microservice displays a "waiting" page, which is meant to indicate to the user that inforamtion is being prepared. Once ready, the microservice will finally show the dashboard itself. This will have a variety of graphs, explained as well, so that the user can interact and understand. There will also be a button to allow the user to enter another game, or change their query selection.

### Data Engineering with Azure

After the user inputs a few parameters, the code will begin its interaction with Azure Databricks. Immediately, the code will check on whether the cluster designated for this project is currently running. If it is not, the code will automatically begin spinning up this cluster. The cluster has a 10 minute inactivity limit for turning off, in attempt to save money. If the cluster is currently running, it will then execute a SQL query. This SQL query will be unique, as it will combine a pre-set skeleton with the user inputs. After query execution, the data will be saved as a ```pandas``` DataFrame. The reason this data structure was chosen was due to its easy compatibility with ```plotly```, which was used to create the interactive graphs. The added benefit of utilizing Azure Databricks was to be able to utilize an effective, and powerful, Infrastructure as Code (IaC) solution.

### Docker

This project is currently contained on DockerHub **ENTER THE DOCKERHUB LINK HERE ONCE ITS CREATED**. The Dockerfile, currently located in the root level of this repository, is used to make the DockerImage for the entire microservice. From here, the container is pushed to Azure Web App, so that the entire dashboard can be deployed to a public endpoint. With this public endpoint, it becomes easier for anyone to access the dashboard. The utilization of Docker and Azure Web App was paired with the thought of, once the pipeline was established, the entire dashboard becomes easily scaleable.

### GitHub Actions

The group implemented a GitHub Actions to promote a CI/CD pipeline. The gates that were established was utilizing ```PyLint``` to lint the code, ```Black``` to format the code, a tester to ensure that all packages were properly downloaded from the ```requirements.txt``` file, and ```PyTest``` as our tester. The group did not stop until it was ensured that everything was passed, functioning properly, and looked presentable. This projects GitHub badges are shown above.

### Load Testing and Quantitative Assessment

As this application could forsee a future with mutliple users, the group decided to load test. Utilizing ```locust```, the group was able check how many users the application can withstand before failing. Setting a maximum of 2,000 users attempting to utilize the microservice, the results below show how successful the entire project was of withstanding a large amount of incoming traffic. The code displaying the behaviors each of these users did is represented in ```locustfile.py```.

{HAVE A PHOTO HERE OF LOCUST AFTER WE ARE DONE}

Along with withstanding a large amount of users, the team felt it would be a success if the average latency per request was anywhere below a minute. This is under the condition that the cluster was already spun, as it takes ~5 minutes for the cluster to initially be created, which severely impacts the latency. The following graph below shows the success.

{HAVE A PHOTO HERE OF THE LATENCY GRAPH IN LOCUST}

### How to Run the Project

A big question remains: How can one run this application locally? To run, all that needs to be performed are the following steps.

1. Clone this respository
2. Run ```make install```, to get the Python version, as well as all the packages, running on your local device
3. Either run directly from ```app.py```, or utilize the ```Flask``` CLI, and set ```FLASK_APP=app.py```, then run ```flask run```

From there, you will get to see the entire application from local. This is also due to the .devcontainer configuration that was downloaded from the repository, which allows easy use of this application within GitHub Codespaces.

### Limitations and Potential Improvement

There were many limitations that occured throughout the creation of this project. First is related to the actual data collection itself. Steam's API is limited on how many requests can be made, where each review is a request to their system. As a result, after a good amount of reviews, the system would place the request under a "timeout" that lasted ~5 minutes. This made data collection a long and strenuous process, as well as made it time consuming to request the reviews for more popular games.

The second limitation came with the reviews themselves. Sometimes the reviews had nothing to do with the games themselves, and were used to make a joke. Other times, the review would express immense distaste for the game before labeling their review as positive. A large number of reviews were labelled as being in English and were not, either being written in another language or being full of emojis. 

A third limitation is how slow Azure Databricks takes to spin up a cluster. As a result, if the user is the first person to attempt to access the cluster in ~10 minutes, it will take almost 5 minutes to start up again. This overhead time is paired with the time necessary to perform the other analysis, which could take longer than what any user is willing to wait. Keeping the cluster constantly on is not a valid solution to this problem, as the cost will quickly pile-up. While these issues are tedious, deviating from Azure Databricks as a whole becomes another issue has many decisions were made surrounding the available tools Azure provides.

From this point, there are a few potential improvements. One is moving the entire project infrastructure to AWS, which comes with pros and cons. While it will speed-up the entire latency time, it will also take significant time to learn the tools AWS uses, and port everything over. The second potential improvement is the add a sentiment analysis feature, which will provide a second interactable feature for users. As well, it can be used to derive a more accurate sentiment on the reviews under a game, which was highlighted earlier as a big issue.

### Utilization of AI Pair Programming Tools

Throughout this project, many team members utilized a variety of different AI Pair Programming tools, the two main tools being ChatGPT and GitHub Copilot. These tools were primarily used for debugging purposes, as working with ```Flask``` and HTML as a whole was entirely new for many team members. These tools were able to explain many of the bugs, and offer insight on how one could potentially implement the wishes of the team into HTML.

The other main use was to explain the interaction between any python file and the project cluster/delta table. This is because utilizing the Databricks REST API was initially difficult, and many approaches were found online with little success. These tools were able to offer a single method, which was then suplemented with the implemetation needed for this project. 

### Architectural Diagram

![image](https://github.com/Ninsta22/Steam-Review-Dashboard/assets/55768636/c568a469-cd79-4131-bdbb-a9810b9e7d05)

### Demo Video

{PLACE DEMO VIDEO HERE}
