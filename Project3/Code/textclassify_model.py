# STEP 1: rename this file to textclassify_model.py

# feel free to include more imports as needed here
# these are the ones that we used for the base model
import numpy as np
import sys
import nltk
from collections import Counter
import math
import os
import pandas as pd




#def valid_down_load():
#    if os.path.exists("./sentiment/vader_lexicon.zip/vader_lexicon/vader_lexicon.txt"):
#        print("Sentiment dictionary is existed.")
#    else:
#        print("Downloading the dictionary, Please wait.")
#        nltk.download('vader_lexicon', "./")
#        print("Download accomplished.")

"""
Your name and file comment here:
"""


"""
Cite your sources here:
"""

"""
Implement your functions that are not methods of the TextClassify class here
"""
def generate_tuples_from_file(training_file_path):
  """
  Generates tuples from file formated like:
  id\ttext\tlabel
  Parameters:
    training_file_path - str path to file to read in
  Return:
    a list of tuples of strings formatted [(id, example_text, label), (id, example_text, label)....]
  """
  f = open(training_file_path, "r", encoding="utf8")
  listOfExamples = []
  for review in f:
    if len(review.strip()) == 0:
      continue
    dataInReview = review.split("\t")
    for i in range(len(dataInReview)):
      # remove any extraneous whitespace
      dataInReview[i] = dataInReview[i].strip()
    t = tuple(dataInReview)
    listOfExamples.append(t)
  f.close()
  return listOfExamples


def make_lex_dict():
#    lexicon_file = "./sentiment/vader_lexicon.zip/vader_lexicon/vader_lexicon.txt"
    """
    Obtain the sentiment dictionary from Vader
    Return: the dictionary 
    
    """
    lexicon_file = "./vader_lexicon.txt"
    lexicon_file = nltk.data.load(lexicon_file)
    lex_dict = {}
    for line in lexicon_file.split("\n"):
        (word, measure) = line.strip().split("\t")[0:2]
        lex_dict[word] = float(measure)   
    return lex_dict



def precision(gold_labels, predicted_labels):
  """
  Calculates the precision for a set of predicted labels give the gold (ground truth) labels.
  Parameters:
      gold_labels (list): a list of labels assigned by hand ("truth")
      predicted_labels (list): a corresponding list of labels predicted by the system
  Returns: double precision (a number from 0 to 1)
  """
  true_positive = 0
  false_positive = 0
  false_negative = 0
  true_negative = 0
  precision_score = 0
  for gold, predict in zip(gold_labels, predicted_labels):
      if gold == "1" and predict == "1":
          true_positive += 1
      if gold == "0" and predict == "1":
          false_positive += 1
      if gold == "1" and predict == "0":
          false_negative += 1
      if gold == "0" and predict == "0":
          true_negative += 1
  precision_score = true_positive / (true_positive + false_positive)
  return precision_score


def recall(gold_labels, predicted_labels):
  """
  Calculates the recall for a set of predicted labels give the gold (ground truth) labels.
  Parameters:
      gold_labels (list): a list of labels assigned by hand ("truth")
      predicted_labels (list): a corresponding list of labels predicted by the system
  Returns: double recall (a number from 0 to 1)
  """
  true_positive = 0
  false_positive = 0
  false_negative = 0
  true_negative = 0
  recall_score = 0
  for gold, predict in zip(gold_labels, predicted_labels):
      if gold == "1" and predict == "1":
          true_positive += 1
      if gold == "0" and predict == "1":
          false_positive += 1
      if gold == "1" and predict == "0":
          false_negative += 1
      if gold == "0" and predict == "0":
          true_negative += 1
  recall_score = true_positive / (true_positive + false_negative)
  return recall_score

def f1(gold_labels, predicted_labels):
  """
  Calculates the f1 for a set of predicted labels give the gold (ground truth) labels.
  Parameters:
      gold_labels (list): a list of labels assigned by hand ("truth")
      predicted_labels (list): a corresponding list of labels predicted by the system
  Returns: double f1 (a number from 0 to 1)
  """

  recall_score = recall(gold_labels, predicted_labels)
  precision_score = precision(gold_labels, predicted_labels)
  if (recall_score == 0 and precision_score == 0):
      return 0
  f1_score = (2 * recall_score * precision_score) / (precision_score + recall_score)

  return f1_score



"""
implement your TextClassify class here
"""
class TextClassify:

  dic = {}

  def __init__(self):
    # do whatever you need to do to set up your class here

    pass

  def train(self, examples):
    """
    Trains the classifier based on the given examples
    Parameters:
      examples - a list of tuples of strings formatted [(id, example_text, label), (id, example_text, label)....]
    Return: None
    """
    
    self.list_id = [x[0] for x in examples]
    self.list_review = [x[1] for x in examples]
    self.list_label = [x[2] for x in examples]
    self.vocabulary = []
    for i in range(len(self.list_review)):
        x = self.list_review[i].split()
        for j in x:
            if j not in self.vocabulary:
                self.vocabulary.append(j)
#   计算｜V｜大小
#   Calculate the｜V｜number
    self.vocabulary_size = len(self.vocabulary)

    
    
  #统计lable个数、每种label对应的文件数量；
    self.label_count = Counter(self.list_label)
  #每个lable对应的词数量
    labels = list(self.label_count.keys())

    label_words = {}
    for label in labels:
      label_words[label] = []
    for line in examples:
      for j in range(0,len(labels)):
        if line[2] == labels[j]:
          label_words[labels[j]].extend(line[1].split())
          continue

    
    self.label_words_count = {}
    for label in label_words:
      self.label_words_count[label] = Counter(label_words[label])
    
    

  def score(self, data):
    """
    Score a given piece of text
    Parameters:
      data - str like "I loved the hotel"
    Return: dict of class: score mappings
    """
    self.score_map = {}
    sentence_split = data.split()
    result = 1
    for key, value in self.label_count.items():
      if key not in self.score_map:
        self.score_map[key] = value / len(self.list_label)
    for key, value in self.label_words_count.items():
        for j in sentence_split:
            if j not in self.vocabulary:
                numerator = 1
                denominator = 1
                result *= (numerator / denominator)
            if j in self.vocabulary:
                numerator = self.label_words_count[key][j] + 1
                denominator = sum(self.label_words_count[key].values()) + self.vocabulary_size
                result *= (numerator / denominator)
        self.score_map[key] *= result
        result = 1
    return self.score_map
    


  def classify(self, data):
    """
    Label a given piece of text
    Parameters:
      data - str like "I loved the hotel"
    Return: string class label
    """
    
    self.result = self.score(data)
    max_key = max(self.result, key=self.result.get)
    return max_key
    
    pass

  def featurize(self, data):
    """
    we use this format to make implementation of part 1.3 more straightforward and to be 
    consistent with what you see in nltk
    Parameters:
      data - str like "I loved the hotel"
    Return: a list of tuples linking features to values
    for BoW, a list of tuples linking every word to True [("I", True), ("loved", True), ("it", True)]
    """
    pass

  def __str__(self):
    return "Naive Bayes - bag-of-words baseline"


  def total_score_info(self, training, testing):
    """
    Get the score information (precision, recall and F1)
    Parameters: 
        training - training data file
        testing - testing data file
    Return: a tuple with these three score information
    
    """
    examples = generate_tuples_from_file(training)
    test_data = generate_tuples_from_file(testing)
    self.train(examples)
    predict_labels = []
    test_line_review = [x[1] for x in test_data]
    ground_truths = [x[2] for x in test_data]
    for data in test_line_review:
        predict_labels.append(self.classify(data))
    precision_score = precision(ground_truths, predict_labels)
    recall_score = recall(ground_truths, predict_labels)
    f1_score = f1(ground_truths, predict_labels)
      
    return (precision_score, recall_score, f1_score)
  

class TextClassifyImproved:

  def __init__(self):
    pass


  def add_feature(self, text):
    """
    Add features into matrix for Logistic Regression
    Parameters:
        Customers' review of the hotel.
    Return: A matrix of features
    """
    pos_feature = []
    neg_feature = []
    score_feature = []
    word_bag_feature = []

    for line in text:
      word_tokens = line.split(" ")
      filtered_sentence = word_tokens
      total_word = len(filtered_sentence)
      score = 0
      pos_num = 0
      neg_num = 0
      dic = make_lex_dict()
      for word in filtered_sentence:
          if word in dic.keys():
              score += dic[word]
              if dic[word] > 0:
                  pos_num += 1
              else:
                  neg_num += 1
      
      pos_feature.append(pos_num)
      neg_feature.append(neg_num)
      word_bag_feature.append(total_word)
      score_feature.append(score)
    feature_matrix = np.array((pos_feature, neg_feature, score_feature, word_bag_feature)).T
#    temp = feature_matrix
#    x = pd.DataFrame(temp, columns = ["pos", "neg", "score", "length"])
#    print(x)
    return feature_matrix

  def sigmoid(self, z):
      """
      The sigmoid function that calculate for Logistic Regression
      Return: The value of the sigmoid
      
      """
      
      return 1 / (1 + np.exp(-z)) 


  def stoc_grad_ascent(self, dataMatIn, classLabels):
     """
     Get the weights of the Logistic Regression
     
     Parameter: 
         dataMatin - the features that are added for the logistic regression in matrix
         classLabels - the label for each reviews
     Return: the weights of the logistic regression
     
     """
     m, n = np.shape(dataMatIn)
     alpha = 0.01
     weights = np.ones(n)
     for i in range(m):
         h = self.sigmoid(sum(dataMatIn[i] * weights))  
         error = classLabels[i] - h
         weights = weights + alpha * error * dataMatIn[i]
     return weights



  def train(self, examples):
    """
    Trains the classifier based on the given examples
    Parameters:
      examples - a list of tuples of strings formatted [(id, example_text, label), (id, example_text, label)....]
    Return: None
    """
    list_review = [x[1] for x in examples]
    list_label = [int(x[2]) for x in examples]

    feature_matrix = self.add_feature(list_review)
    dataMatIn = np.insert(feature_matrix, 0, 1, axis=1) 
    labelMat = np.array(list_label)
    self.weights = self.stoc_grad_ascent(dataMatIn, labelMat)




  def score(self, data):
    """
    Score a given piece of text
    you’ll compute e ^ (log(p(c)) + sum(log(p(w_i | c))) here
    
    Parameters:
      data - str like "I loved the hotel"
    Return: dict of class: score mappings
    return a dictionary of the values of P(data | c)  for each class, 
    as in section 4.3 of the textbook e.g. {"0": 0.000061, "1": 0.000032}
    """
    list_review = []
    score_dict = {}
    list_review.append(data)
    feature_matrix = self.add_feature(list_review)
    sum_prob = self.weights[0]

    for i in range(0,4):
        sum_prob += feature_matrix[0][i] * self.weights[i+1]
        
    sigma = self.sigmoid(sum_prob)
    score_dict["1"] = sigma
    score_dict["0"] = 1 -  sigma  
    return score_dict


  def classify(self, data):
    """
    Label a given piece of text
    Parameters:
      data - str like "I loved the hotel"
    Return: string class label
    """
    result = self.score(data)
    max_key = max(result, key = result.get)
    return max_key
    
    pass

  def featurize(self, data):
    """
    we use this format to make implementation of part 1.3 more straightforward and to be 
    consistent with what you see in nltk
    Parameters:
      data - str like "I loved the hotel"
    Return: a list of tuples linking features to values
    for BoW, a list of tuples linking every word to True [("I", True), ("loved", True), ("it", True)]
    """
    
    pass

  def total_score_info(self, training, testing):
    """
    Get the score information (precision, recall and F1)
    Parameters: 
        training - training data file
        testing - testing data file
    Return: a tuple with these three score information
    
    """
    examples = generate_tuples_from_file(training)
    test_data = generate_tuples_from_file(testing)
    self.train(examples)
    predict_labels = []
    test_line_review = [x[1] for x in test_data]
    ground_truths = [x[2] for x in test_data]
    for data in test_line_review:
        predict_labels.append(self.classify(data))
    precision_score = precision(ground_truths, predict_labels)
    recall_score = recall(ground_truths, predict_labels)
    f1_score = f1(ground_truths, predict_labels)
     
    return (precision_score, recall_score, f1_score)


  def __str__(self):
    return "Logistic Regression"



def main():

  training = sys.argv[1]
  testing = sys.argv[2]

  classifier = TextClassify()
  print(classifier)
  # do the things that you need to with your base class
  (precision_score, recall_score, f1_score) = classifier.total_score_info(training, testing)
  print("Precision score: ", precision_score)
  print("Recall score: ", recall_score)
  print("F1 score: ", f1_score)
  
  

  # report precision, recall, f1
  
  

  improved = TextClassifyImproved()
  print(improved)
  
  (precision_improve_score, recall_improve_score, f1_improve_score) = improved.total_score_info(training, testing)
  print("Precision score: ", precision_improve_score)
  print("Recall score: ", recall_improve_score)
  print("F1 score: ", f1_improve_score)
  # do the things that you need to with your improved class


  # report final precision, recall, f1 (for your best model)




if __name__ == "__main__":
  if len(sys.argv) != 3:
    print("Usage:", "python textclassify_model.py training-file.txt testing-file.txt")
    sys.exit(1)

  main()
 








