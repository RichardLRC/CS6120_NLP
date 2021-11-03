# STEP 1: rename this file to textclassify_model.py

# feel free to include more imports as needed here
# these are the ones that we used for the base model
import numpy as np
import sys
from collections import Counter
import math

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

  #统计lable个数、每种label对应的文件数量；
  list_label = [x[2] for x in listOfExamples]
  label_count = Counter(list_label)
  #每个lable对应的词数量
  labels = list(label_count.keys())
  label_words = {} #{label0:[word1,word2,^],label1:[]}  {word:{label}}
  lable_words_count = {}
  for label in labels:
    label_words[label] = [ ]
  print(label_words)
  for line in listOfExamples:
    x = line[1].split()
    for j in range(0,len(labels)):
      if line[2] == labels[j]:
        label_words[labels[j]].extend(x)
        continue
  print(label_words)
  for label in label_words.keys():
    lable_words_count[label] = Counter(label_words[label])
  print(lable_words_count)
  return listOfExamples
print(generate_tuples_from_file("minitrain.txt"))
def precision(gold_labels, predicted_labels):
  """
  Calculates the precision for a set of predicted labels give the gold (ground truth) labels.
  Parameters:
      gold_labels (list): a list of labels assigned by hand ("truth")
      predicted_labels (list): a corresponding list of labels predicted by the system
  Returns: double precision (a number from 0 to 1)
  """
  #计算得到标签名称
  labels_count = Counter(gold_labels)

  labelslist = labels_count.keys()

  matrix = [[]]

  #两种类别下的准确率计算
  templist = []
  if len(labels_count.keys()) > 2:
    #每类label
    for labelvalue in labels_count.keys():
      #获取真值为某一标签的列表
      for gold, predict in zip(gold_labels, predicted_labels):
        if labelvalue == gold:
          templist.appenf[predict]
      tempdict = Counter(templist)
      #值存到矩阵
      for temp in tempdict.keys():
        rownum = labelslist.index(labelvalue)
        colnum = labelslist.index(temp)
        matrix[rownum][colnum] = tempdict[temp]/(labels_count[labelvalue]*1.0)
  print(matrix)

  pass


def recall(gold_labels, predicted_labels):
  """
  Calculates the recall for a set of predicted labels give the gold (ground truth) labels.
  Parameters:
      gold_labels (list): a list of labels assigned by hand ("truth")
      predicted_labels (list): a corresponding list of labels predicted by the system
  Returns: double recall (a number from 0 to 1)
  """
  pass

def f1(gold_labels, predicted_labels):
  """
  Calculates the f1 for a set of predicted labels give the gold (ground truth) labels.
  Parameters:
      gold_labels (list): a list of labels assigned by hand ("truth")
      predicted_labels (list): a corresponding list of labels predicted by the system
  Returns: double f1 (a number from 0 to 1)
  """
  pass


"""
Implement any other non-required functions here
"""



"""
implement your TextClassify class here
"""
class TextClassify:


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
    pass

  def score(self, data):
    """
    Score a given piece of text
    Parameters:
      data - str like "I loved the hotel"
    Return: dict of class: score mappings
    """
    #计算在每个类别下的得分
    #循环


    pass

  def classify(self, data):
    """
    Label a given piece of text
    Parameters:
      data - str like "I loved the hotel"
    Return: string class label
    """
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


class TextClassifyImproved:

  def __init__(self):
    pass

  def train(self, examples):
    """
    Trains the classifier based on the given examples
    Parameters:
      examples - a list of tuples of strings formatted [(id, example_text, label), (id, example_text, label)....]
    Return: None
    """
    for line in examples:
      self.list_id = line[0]
      self.list_review = line[1]
      self.list_label = line[2]
    pass

  #统计lable个数、每种label对应的文件数量；
    self.list_label = [x[2] for x in examples]
    label_count = Counter(self.list_label)
  #每个lable对应的词数量
    labels = list(label_count.keys())
    label_words = {}
    lable_words_count = {}
    for label in labels:
      label_words[label] = []
    for line in examples:
      for j in range(0,len(labels)):
        if line[2] == labels[j]:
          label_words[labels[j]].extend(line[1].split())
          continue
    for lable in label_words:
      lable_words_count[label] = Counter(label_words[label])










    label_words = {}

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
    pass

  def classify(self, data):
    """
    Label a given piece of text
    Parameters:
      data - str like "I loved the hotel"
    Return: string class label
    """
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
    return "NAME OF YOUR CLASSIFIER HERE"



def main():

  training = sys.argv[1]
  testing = sys.argv[2]

  classifier = TextClassify()
  print(classifier)
  # do the things that you need to with your base class


  # report precision, recall, f1
  

  improved = TextClassifyImproved()
  print(improved)
  # do the things that you need to with your improved class


  # report final precision, recall, f1 (for your best model)




if __name__ == "__main__":
  if len(sys.argv) != 3:
    print("Usage:", "python textclassify_model.py training-file.txt testing-file.txt")
    sys.exit(1)

  main()
 








