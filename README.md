# Sentiment-Analysis
Sentiment Analysis using Logistic Regression

#Clean the Data 
1. Clean the Data using Regex. 
2. Remove all Punctuations, Numbers, extra space, repetative words, urls(http, https), @mention_name.

#Data Imbalance
1. Because data is imbalance so to handle Data overfitting using sampling technique.
2. Check Model accuracy using Random Over sampler and Random Under Sampler. 
3. After Checking with these techniques then find Random Over Sampler is provide good accuracy. 

#Train Model
1. Use TFidf vectorizing for feature extraction.
2. Logistic Regression for predications.

#After train Model get and predicating class. Get a  0.4810 F1-score. 
