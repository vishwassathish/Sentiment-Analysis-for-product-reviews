# Sentiment-Analysis-for-product-review

1) Loading the data <br/>
  Load the raw data into python lists  
  
2) Process to sentences <br/>
  Convert the raw reviews to sentences  
  
3) Text preprocessing  <br/>
  Tokenize the texts using keras. preprocessing.text module  
  
4) Create Training set and validation set  <br/>
  Randomly pick reviews(which are now converted to lists of tokens. one list per review) from the pool of dataset and assing it as train 
  and test samples.  
  
5) Create word2vec embeddings using GloVe  <br/>
  Use the GloVe dataset to convert each word of the review into a 100 dimension tensor, which is ready to be sent into our model for 
  training.  Get the GloVe Dataset from <b> https://nlp.stanford.edu/projects/glove/ </b>
  
6) Training  <br/>
  Used LSTM network for our model. Used 
  dropout activation function = sigmoid 
  error function =  binary_crossentropy
