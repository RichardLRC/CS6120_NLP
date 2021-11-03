# imports go here
import sys
import random
from collections import Counter
from statistics import mean
from statistics import stdev
"""
Don't forget to put your name and a file comment here
"""


# Feel free to implement helper functions

class LanguageModel:
  # constants to define pseudo-word tokens
  # access via self.UNK, for instance
  UNK = "<UNK>"
  SENT_BEGIN = "<s>"
  SENT_END = "</s>"

  def __init__(self, n_gram, is_laplace_smoothing):
    """Initializes an untrained LanguageModel
    Parameters:
      n_gram (int): the n-gram order of the language model to create
      is_laplace_smoothing (bool): whether or not to use Laplace smoothing
    """
    self.n_gram = n_gram
    self.is_laplace_smoothing = is_laplace_smoothing
    pass


  def train(self, training_file_path):
    """Trains the language model on the given data. Assumes that the given data
    has tokens that are white-space separated, has one sentence per line, and
    that the sentences begin with <s> and end with </s>
    Parameters:
      training_file_path (str): the location of the training data to read

    Returns:
    None
    """
    file = open(training_file_path, "r")
    content = file.read()
    content_list = content.split()
    self.count = Counter(content_list)
    UNK_list = []
    for key, value in self.count.items():
        if value == 1:
            if key == self.SENT_BEGIN:
                continue
            elif key == self.SENT_END:
                continue
            else:
                UNK_list.append(key)

    with open(training_file_path, "r") as f:
        x = ([line.split() for line in f])
    for i in range(len(x)):
        for j in range(len(UNK_list)):
            if UNK_list[j] in x[i]:
                index = x[i].index(UNK_list[j])
                x[i][index] = self.UNK     

    self.total_grams = []
    self.word = []
    if self.n_gram > 1:
        for line in x:
            [self.total_grams.append(tuple(line[i: i+self.n_gram])) for i in range(len(line) - self.n_gram + 1)]
            [self.word.append(tuple(line[i: i+self.n_gram- 1])) for i in range(len(line) - self.n_gram + 2)]
    total_word = []
    if self.n_gram == 1:
        for line in x:
            for i in line:
                total_word.append(i)
                self.unicount = Counter(total_word)
        
        for i in self.unicount:
            self.total_grams.append(i)
            self.word.append(self.unicount[i])

    pass

  def make_ngram (self, n, tokens):
    ngram_list = []
    
    ngrams = zip(*[tokens[i:] for i in range(n)])
    for ngram in ngrams:
        ngram_list.append(ngram)
        
    return ngram_list

  def score(self, sentence):
    """Calculates the probability score for a given string representing a single sentence.
    Parameters:
      sentence (str): a sentence with tokens separated by whitespace to calculate the score of
      
    Returns:
      float: the probability value of the given string for this model
    """
    if (self.n_gram == 1):
        list_frequencies = Counter(self.total_grams)

        list_frequencies_key = list(list_frequencies.keys())
        list_input = sentence.split(" ")
        list_revise = []

        result = 1
        for i in list_input:    
            if (i not in list_frequencies_key):
    
                list_revise.append(self.UNK)
            else:
                list_revise.append(i)

        for i in list_revise:
            if self.is_laplace_smoothing == True:

                probability_numerator = self.unicount.get(i) + 1
                probability_denominator = len(self.total_grams) + sum(self.word)
                result *= (probability_numerator) / (probability_denominator)
            else:
                probability_numerator = self.unicount.get(i)

                probability_denominator = sum(self.word)
                result *= float(probability_numerator) / float(probability_denominator)
    
    if self.n_gram > 1:   
        list_frequencies = Counter(self.total_grams)


        list_frequencies_key = list(list_frequencies.keys())

        list_input = sentence.split(" ")
        list_revise = []
        list_single = [i[0] for i in self.word]
        set_single = set(list_single)
#        print("list_single")
#        print(list_single)
        result = 1

        for i in list_input:    
            if (i not in list_single):
    
                list_revise.append(self.UNK)
            else:
                list_revise.append(i)
        list_of_ngram = self.make_ngram(self.n_gram, list_revise)
#  
        single_result = 1
        for i in list_of_ngram:
            for j in range(0, len(list_revise) - 1):
                current_str = list_revise[j]
                if self.is_laplace_smoothing == True:
                    probability_numerator = list_frequencies[i] + 1
                    probability_denominator = list_single.count(current_str) + len(set_single)
#                    print(list_single.count(current_str))
#                    print(len(set_single))
#            result *= float(probability_numerator) / float(probability_denominator)
                else:
                    probability_numerator = list_frequencies[i]
#                    print(probability_numerator)
                    probability_denominator = list_single.count(current_str)
#            print(probability_numerator)
#            print(probability_denominator)
            single_result = float(probability_numerator) / float(probability_denominator)
            result *= single_result
#        print(result)
                
    return result

  def perplexity(self, test_sequence):
      list_string = test_sequence.split(" ")
      n_begin_string = list_string.count(self.SENT_BEGIN)
      totalN = len(list_string) - n_begin_string
      result = pow(self.score(test_sequence), -1 / totalN)
      return result 
      

  def generate_sentence(self):
    """Generates a single sentence from a trained language model using the Shannon technique.
      
    Returns:
      str: the generated sentence
    """
    sentence = ''
    if self.n_gram == 1:
        sentence_list = []
        sentence += self.SENT_BEGIN
        temp_unicount = self.unicount
        if self.SENT_BEGIN in temp_unicount:
          temp_unicount.pop(self.SENT_BEGIN)
        list_key_unicount = list(temp_unicount.keys())
        list_value_unicount = list(temp_unicount.values())

        next_words = random.choices(list_key_unicount, weights = list_value_unicount,k=30)

        seqlist = [s.split() for s in ' '.join(next_words).split('</s>') if s]

        templist = []
        for seq in seqlist:
            if len(seq)>1:
                templist.append(seq)
        sentence_list = random.choice(templist)
        
        #combine sentence with <s> and </s>
        middle_part = " ".join(sentence_list)
        sentence = self.SENT_BEGIN + " " + middle_part + " " + self.SENT_END
    else:

        flag = True
        sentence_list = []
        lensen = 0

        tranlist = []
        for i in range(self.n_gram-1):
            tranlist.append("<s>")
        trantuple = tuple(tranlist)

        while flag and lensen < 30:
            temp_garm = Counter([item for item in self.total_grams if trantuple == item[:-1]])
            # print(temp_garm)
            keys = list(temp_garm.keys())
            values = list(temp_garm.values())
            # print(keys[:min(5,len(keys))])
            # temptuple = random.choice(keys[:min(5,len(keys))])
            next_words = random.choices(keys, weights=values)
            temptuple = next_words[0]
            tuplestr = " ".join(temptuple)
            trantuple = temptuple[1:]
            if tuplestr.split()[-1] == "</s>":
                break
            else:
                sentence_list.append(tuplestr.split()[-1])
#            print(sentence_list)
            lensen += 1

        middle_part = " ".join(sentence_list)
        sentence = self.SENT_BEGIN + " " + middle_part + " " + self.SENT_END


    return sentence



    
  def generate(self, n):
    """Generates n sentences from a trained language model using the Shannon technique.
    Parameters:
      n (int): the number of sentences to generate
      
    Returns:
      list: a list containing strings, one per generated sentence
    """
    list_sentence = []
    for i in range(n):
        list_sentence.append(self.generate_sentence())
    return list_sentence


def main():
  # TODO: implement
      training_path = sys.argv[1]
      testing_path1 = sys.argv[2]
      testing_path2 = sys.argv[3]
      lm =  LanguageModel(1, True)
      lm.train(training_path)
      unigram_sentence_set = []
      unigram_sentence_probability_set = []
      with open(testing_path1) as file:
          for line in file:
              unigram_sentence_set.append(line.rstrip())
      for i in range(len(unigram_sentence_set)):
          unigram_sentence_probability_set.append(lm.score(unigram_sentence_set[i]))

      print("Model: unigram, laplace smoothed")
      print("50 Sentences: ")
      unigram_sentence_list = lm.generate(50)
      print(*unigram_sentence_list, sep = "\n")
      print("\n")
      print("test corpus: hw2-test.txt")
      print("# of test sentences: 100")
      print("Average probability: ", mean(unigram_sentence_probability_set))
      print("Standard deviation: ", stdev(unigram_sentence_probability_set))

#      
#      
      lm1 =  LanguageModel(1, True)
      lm1.train(training_path)
      unigram_sentence_my_set = []
      unigram_sentence_probability_my_set = []
      with open(testing_path2) as file:
          for line in file:
              unigram_sentence_my_set.append(line.rstrip())
      for i in range(len(unigram_sentence_my_set)):
          unigram_sentence_probability_my_set.append(lm1.score(unigram_sentence_my_set[i]))
      print("\n")
      print("test corpus: hw2-my-test.txt")
      print("# of test sentences: 100")
      print("Average probability: ", mean(unigram_sentence_probability_my_set))
      print("Standard deviation: ", stdev(unigram_sentence_probability_my_set))
      
      
      lm2 =  LanguageModel(2, True)
      lm2.train(training_path)
      bigram_sentence_set = []
      bigram_sentence_probability_set = []
      with open(testing_path1) as file:
          for line in file:
              bigram_sentence_set.append(line.rstrip())
      for i in range(len(bigram_sentence_set)):
          bigram_sentence_probability_set.append(lm2.score(bigram_sentence_set[i]))
      print("\n")
      print("Model: bigram, laplace smoothed")
      print("50 Sentences: ")
      bigram_sentence_list = lm.generate(50)
      print(*bigram_sentence_list, sep = "\n")
      print("\n")
      print("test corpus: hw2-test.txt")
      print("# of test sentences: 100")
      print("Average probability: ", mean(bigram_sentence_probability_set))
      print("Standard deviation: ", stdev(bigram_sentence_probability_set))
#      
#      
      lm3 =  LanguageModel(2, True)
      lm3.train(training_path)
      bigram_sentence_my_set = []
      bigram_sentence_probability_my_set = []
      with open(testing_path2) as file:
          for line in file:
              bigram_sentence_my_set.append(line.rstrip())
      for i in range(len(bigram_sentence_my_set)):
          bigram_sentence_probability_my_set.append(lm3.score(bigram_sentence_my_set[i]))
      print("\n")
      print("test corpus: hw2-my-test.txt")
      print("# of test sentences: 100")
      print("Average probability: ", mean(bigram_sentence_probability_my_set))
      print("Standard deviation: ", stdev(bigram_sentence_probability_my_set))
      print("\n")
      
      
            
      lm_per = LanguageModel(1, True)
      lm_per.train(training_path)
      with open(testing_path1) as myfile:
          head = [next(myfile).rstrip() for x in range(10)]
      list_string = ' '.join(head)
      print("Perplexity for 1-grams:")
      print("hw2-test.txt: ", lm_per.perplexity(list_string))
      
          
      lm_per_my = LanguageModel(1, True)
      lm_per_my.train(training_path)
      with open(testing_path2) as myfile:
          head = [next(myfile).rstrip() for x in range(10)]
      list_string = ' '.join(head)
      print("hw2-my-test.txt: ", lm_per_my.perplexity(list_string))
            
      print("\n")
      lm_per_2 = LanguageModel(2, True)
      lm_per_2.train(training_path)
      with open(testing_path1) as myfile:
          head = [next(myfile).rstrip() for x in range(10)]
      list_string = ' '.join(head)    
      print("Perplexity for 2-grams:")
      print("hw2-test.txt: ", lm_per_2.perplexity(list_string))
            
      lm_per_2_my = LanguageModel(2, True)
      lm_per_2_my.train(training_path)
      with open(testing_path2) as myfile:
          head = [next(myfile).rstrip() for x in range(10)]   
      list_string = ' '.join(head)   
      print("hw2-my-test.txt: ", lm_per_2_my.perplexity(list_string))
      


## self-test for high n-gram, Trigram. 
#      print("\n")
#      lm4 =  LanguageModel(3, True)
#      lm4.train("/Users/richardli/Desktop/NEU/CS6120/Homework2_1version/Training data/Full data/berp-training-tri.txt")      
#      print("Model: Trigram, laplace smoothed")
#      print("50 Sentences: ")
#      trigram_sentence_list = lm4.generate(50)
#      print(*trigram_sentence_list, sep = "\n")
#      trigram_sentence_my_set = []
#      trigram_sentence_probability_my_set = []
#      with open("/Users/richardli/Desktop/NEU/CS6120/Homework2_1version/Testing data/hw2-test-tri.txt") as file:
#          for line in file:
#              trigram_sentence_my_set.append(line.rstrip())
#      for i in range(len(trigram_sentence_my_set)):
#          trigram_sentence_probability_my_set.append(lm4.score(trigram_sentence_my_set[i]))
#      print("\n")
#      print("test corpus: berp-training-tri.txt")
#      print("# of test sentences: 100")
#      print("Average probability: ", mean(trigram_sentence_probability_my_set))
#      print("Standard deviation: ", stdev(trigram_sentence_probability_my_set))
 
## self-test for high n-gram, Fourgram.      
#      print("\n")
#      lm5 =  LanguageModel(4, True)
#      lm5.train("/Users/richardli/Desktop/NEU/CS6120/Homework2_1version/Training data/Full data/berp-training-four.txt")      
#      print("Model: Fourgram, laplace smoothed")
#      print("50 Sentences: ")
#      fourgram_sentence_list = lm5.generate(50)
#      print(*fourgram_sentence_list, sep = "\n")
#      fourgram_sentence_my_set = []
#      fourgram_sentence_probability_my_set = []
#      with open("/Users/richardli/Desktop/NEU/CS6120/Homework2_1version/Testing data/hw2-test-four.txt") as file:
#          for line in file:
#              fourgram_sentence_my_set.append(line.rstrip())
#      for i in range(len(fourgram_sentence_my_set)):
#          fourgram_sentence_probability_my_set.append(lm5.score(fourgram_sentence_my_set[i]))
#      print("\n")
#      print("test corpus: berp-training-four.txt")
#      print("# of test sentences: 100")
#      print("Average probability: ", mean(fourgram_sentence_probability_my_set))
#      print("Standard deviation: ", stdev(fourgram_sentence_probability_my_set))
      
      
if __name__ == '__main__':
    
  # make sure that they've passed the correct number of command line arguments
#  if len(sys.argv) != 4:
#    print("Usage:", "python lm.py training_file.txt testingfile1.txt testingfile2.txt")
#    sys.exit(1)

  main()

