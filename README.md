# Stock-Prediction-with-daily-news-headlines

Besides the historical stock data, stock fluctuation is also highly correlated with daily news headlines. We propose to build the machine learning classifier and neural networks for investors in order to have a better prediction of stock trends with the news headlines. Accuracy is used as the indicator of our model performance. We found that the TF-IDF Vectorizer + Randomforest Classifier and Recurrent Neural Network has the highest accuracy of 85.71%.

In part one, we tried out different combinations of the vectorizers and classifiers for training. The two vectorization tools are TF-IDF vectorizer and CountVectorizer. The two classification models are Random Forest Classifier and Naive Bayes Classifier. We implemented them by using the scikit-learn packages. In the second part, we used the torchtext package for loading data and used the pre-trained word embedding vector, which is ‘glove.6B.100d’ for the text vectorization. We used Pytorch to build our neural network for training. Our implementation includes the Recurrent Neural Network and Convolutional Neural Network.

Models Performance Comparison: (from high accuracy to low)
TF-IDF Vectorizer + Random Forest Classifier (TFIDF+RF): 0.8571
Recurrent Neural Network: 0.8571
Convolutional Neural Network: 0.8519
TF-IDF Vectorizer + Naive Bayes Classifier (TFIDF+NB): 0.8518
Countvectorizer + Random Forest Classifier (CV+RF): 0.8492
Countvectorizer + Naive Bayes Classifier (CV+NB): 0.8412

TFIDF+RF and RNN have the highest accuracy 85.71%.

Prior Work / References:
[1] Gidofalvi, G., & Elkan, C. (2001). Using news articles to predict stock price movements. Department of Computer Science and Engineering, University of California, San Diego.

[2] Pennington, J., Socher, R., & Manning, C. D. (2014, October). Glove: Global vectors for word representation. In Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP) (pp. 1532-1543).

[3] https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html

[4] Zhang, L., Wang, S., & Liu, B. (2018). Deep learning for sentiment analysis: A survey. Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery, 8(4), e1253.

[5] Sun, J. (2016, August). Daily News for Stock Market Prediction, Version 1. Retrieved [Date You Retrieved This Data] from https://www.kaggle.com/aaron7sun/stocknews.

[6] https://pytorch.org/docs/stable/generated/torch.nn.RNN.html

[7] https://d2l.ai/chapter_natural-language-processing-applications/sentiment-analysis-cnn.html

[8] course material: NLP_Live_Session_Notebook.ipynb
