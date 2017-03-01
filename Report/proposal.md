# NewsToMarket

## Abstract
Big data is one of the more fashionable catch phrases of our time but unlike many such fads there is something valuable at the heart of it. We're, through our interactions on the internet and web, gathering information on what we all think and do in unprecedented volumes. The difficulty, as with any new source of information, is in knowing how to actually make use of it. If you like, the task is to extract the information, the useful part, from the flood of data that surrounds it. One such attempt gives us the interesting news that we might be able to predict stock market movement and direction by feeding the machine learning algorithms with the top 25 news headlines of the specific day.

This  project aims to  shows  that  short-term  stock  index  movements  can  be predicted  using  unstructured financial cues like news  articles. Given an historical corpus of eod of day stock  index  movement and financial news, we intend to build a model to predict current eod index movement. Since we will essentially be predicting whether the index goes up or down given the news data; our model fits in the realm of classification problem and we therefore intend to build a series of classifiers using Logistic Regression, Gaussian Naive Bayes, Random Forest and so on. A market sentiment has direct bearings on the market index movement; we therefore intend to apply sentiment analysis on the news corpus to identify the positive or negative sentiment and integrate the results in the model. Further we also want to see if presence of certain words can have any influence on the market for which we plan to apply and use topic modeling.

## Intro
According  to  the   “efficient   market   hypothesis”,   in   financial   markets   profit   opportunities are exploited as soon as they arise, hence stock prices follow a random walk and are extremely difficult to predict. However, in  the  financial  market  setting  the  task  is  rather  to  generate  profitable  action signals (buy and sell) than to accurately predict future values of a time series.  A usually less successful technical analysis tries to predict  future  prices  based  on  past  prices,  whereas  fundamental  analysis  tries  to  base  predictions  on  factors  in  the  real  economy  (e.g.  inflation,  trading  volume,  organizational  changes  in  the  company,  demand  for  products  or  services  offered  by  the company).  As textual data (news articles) became available on the web, a   new   source   of   indicators   appear,   which   potentially   could   contain   useful    information for fundamental analysis. The objective of this project is to analyze and extract such information, and derive numerical indicators from financial text. 


## Motivation
The basic motivation of this project is to apply machine learning techniques in real life problem to provide insights, and make a good data-oriented decision based on predictions given by the machine learning algorithm.
Combine the concepts of machine learning and text analytics to gain better insights both from market data and text data (i.e. News and Forums).
Expected prediction results are the stock movements in the near future with the input of past and current data.

## Data Description
There are three data files in CSV format to be used for this machine learning project

* RedditNews.csv: Historical news headlines from Reddit WorldNews Channel. It has a total of 73,608 count of headlines from the period 06/08/2008 to 01/07/2016. Only top 25 headlines are considered for each date.

* DJIA_tables.csv: Daily rates of Dow Jones Industrial Average (DJIA) from date range 08/08/2008 to 01/07/2016.  A total of 7 Columns containing the date, opening stock price, day highest price, day lowest stock price, closing stock price, volume and adjusted close price.

* Combined_News_DJIA.csv: Combined dataset of both RedditNews and DJIA tables dataset. A total of 1989 dates, with 25 columns reflecting the ranking of headline in each date(row). Each row has 25 top headlines of each date. Column B, “Label’ , reflects the binary classification as below.
"1" when DJIA Adjusted Close value rose or stayed as the same;

"0" when DJIA Adjusted Close value decreased.

## Objective
We will be using the newspaper contents to build a classification model in order to predict the stock market movement
i.e. whether the stock market will be closing at a high, low or will remain unchanged at the end of the day.

Some of our objectives are as follows:
* Using the NLTK toolkit in order to convert the newspaper content into bag of words which will be then used to predict the stock market movement.

* Using techniques such as NGram which are basically set of co-occurring words within a given window, which will then be used to predict the stock market movement

* Topic modeling will be done in order to find out suitable topics/label of the news content, and will then be used to predict the stocks.

* Lastly we will be doing a sentiment analysis on the newspaper content to find out whether positive or negative news influence the stock market movement.

## Approach
The set of 25 news for each day needs to be transformed to a format suitable for performing textual analysis and applying predictive model. 

We will be using both "BOW" and "n-gram" technique to model the data and perform a comparative performance analysis. 

1. Prepare the corpus
	* Apply tokenization to separate the English tokens 
	* Stopwords removal and lemmatization 
	* Dictionary creation( bow , n-gram ) 
	* Tf-idf weighting 

2. Apply Logistic, GNB models on the corpus and up with an accuracy value 

3. Perform topic modeling 
	* Apply LDA( Latent Dirichlet Allocation ) to come up with a representative topic for each doc 
	* Apply elbow analysis on k-means clustering to come up with an optimal number of topics

4. Apply sentiment analysis to extract the positive or negative sentiment of the combined docs per day and use this along with lda topics above to see if there are any improvements 

## References
* https://www.researchgate.net/publication/228892903_Using_news_articles_to_predict_stock_price_movements

* http://www.pnas.org/content/111/32/11600.full.pdf
 
* https://www.kaggle.com/aaron7sun/stocknews
