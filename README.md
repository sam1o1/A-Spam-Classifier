# Building a Spam Classifier
![Udacity](https://upload.wikimedia.org/wikipedia/commons/3/3b/Udacity_logo.png)

This Project is part of   [_Udacity NLP Nanodegree_](https://www.udacity.com/course/natural-language-processing-nanodegree--nd892)

![Spam](https://github.com/sam1o1/A-Spam-Classifier/blob/main/Spam%20Classifier/images/c4183680-fb99-11e9-8191-d7c5dfb6a11e.png?raw=true)

Spam detection is one of the major applications of Machine Learning in the interwebs today. Pretty much all of the major email service providers have spam detection systems built in and automatically classify such mail as 'Junk Mail'.
In this mission we will be using the Naive Bayes algorithm to create a model that can classify  [dataset](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)
SMS messages as spam or not spam, based on the training we give to the model. It is important to have some level of intuition as to what a spammy text message might look like.
Usually they have words like 'free', 'win', 'winner', 'cash', 'prize' and the like in them as these texts are designed to catch your eye and in some sense tempt you to open them. Also, spam messages tend to have words written in all capitals and also tend to use a lot of exclamation marks. To the recipient, it is usually pretty straightforward to identify a spam text and our objective here is to train a model to do that for us!

Being able to identify spam messages is a binary classification problem as messages are classified as either 'Spam' or 'Not Spam' and nothing else. Also, this is a supervised learning problem, as we will be feeding a labelled dataset into the model, that it can learn from, to make future predictions.
## Installations 
 - [Python](https://www.python.org/downloads/release/python-364/)
 -   [NumPy](http://www.numpy.org/)
 -   [pandas](http://pandas.pydata.org/)
 -   [scikit-learn](http://scikit-learn.org/0.17/install.html)  (v0.17)
 -   [Matplotlib](http://matplotlib.org/)
 
You will also need to have software installed to run and execute a  [Jupyter Notebook](http://ipython.org/notebook.html).
If you do not have Python installed yet, it is highly recommended that you install the  [Anaconda](http://continuum.io/downloads)  distribution of Python, which already has the above packages and more included.
## Project Motivation 
Machine learning is used to solve real issues that humans can spend forever trying to solve them. Saving time, money and efforots are exactly what motivated me to work on this project. Building a Supervised Machine learning model that is capable of Classifying Spammy emails help clients to focus more on important emails rather than getting confused by the tedious ads that always come to the inbox.
## File Descriptions

`Bayesian_Inference.ipynb`: it includes :

* Step 0: Introduction to the Naive Bayes Theorem
* Step 1.1: Understanding our dataset
* Step 1.2: Data Preprocessing
* Step 2.1: Bag of Words(BoW)
* Step 2.2: Implementing BoW from scratch
* Step 2.3: Implementing Bag of Words in scikit-learn
* Step 3.1: Training and testing sets
* Step 3.2: Applying Bag of Words processing to our dataset.
* Step 4.1: Bayes Theorem implementation from scratch
* Step 4.2: Naive Bayes implementation from scratch
* Step 5: Naive Bayes implementation using scikit-learn
* Step 6: Evaluating our model
* Step 7: Conclusion

`smsspamcollection`: The dataset used. 


