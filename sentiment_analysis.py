import tensorflow as tf
import nltk
import numpy as np
import random
import pickle #To save and restore the model at some point
from collections import Counter #To keep count of certain things
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
nm_lines = 10000000

def create_lexicon(pos,neg):
    lexicon = []
    for fi in [pos,neg]:
        with open(fi,'r') as f:
            contents = f.readlines()
            for l in contents[:nm_lines]:
                all_words = word_tokenize(l.lower())
                lexicon += list(all_words)
    # Lemmatize the lexicon
    lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
    w_counter = Counter(lexicon) 
    # What this does is that creates a dictionary of the word and their counts in the lexicon 
    # For ex : {'and':3444,'this',21}
    l2 = []
    for w in w_counter:
        if 1000 > w_counter[w] > 50:
            l2.append(w)
    # We don't want words that are very common, so we are taking words whose count is greater than 50 and less than 1000
    # So l2 is our final lexicon
    print(len(l2))
    return l2

def sample_handling(sample,lexicon,classification):
    featureset = []
    with open(sample,'r') as f:
        contents = f.readlines()
        for l in contents[:nm_lines]:
            current_word = word_tokenize(l.lower())
            current_word = [lemmatizer.lemmatize(i) for i in current_word]
            features = np.zeros(len(lexicon))
            for word in current_word:
                if word.lower() in lexicon:
                    index_value = lexicon.index(word.lower())
                    features[index_value] += 1
            
            features = list(features)
            featureset.append([features,classification])
            
    return featureset
                    
def create_feature_sets_and_labels(pos,neg,test_size=0.1):
    lexicon = create_lexicon(pos,neg)
    features = []
    features += sample_handling(pos,lexicon,[1,0])
    features += sample_handling(neg,lexicon,[0,1])
    
    random.shuffle(features)
    features = np.array(features)
    
    testing_size = int(test_size*len(lexicon))
    
    train_x = list(features[:,0][:-testing_size])
    train_y = list(features[:,1][:-testing_size])
    
    test_x = list(features[:,0][-testing_size:])
    test_y = list(features[:,1][-testing_size:])
    
    return train_x,train_y,test_x,test_y

if __name__ == '__main__':
    train_x,train_y,test_x,test_y = create_feature_sets_and_labels('pos.txt','neg.txt')
    with open('sentiment_set.pickle','wb') as f:
        pickle.dump([train_x,train_y,test_x,test_y],f)
    
    
    
    
                    
    
    


    











































# Function of word_tokenize: It seperates all the words in a sentence in a list
# Ex: 'I pulled the chair upto the table' gets converted to [I,pulled,the,chair,upto,the,table]

'''
    The goal of both stemming and lemmatization is to reduce inflectional forms and sometimes derivationally related forms of a word to a common base form. For instance:
    am, are, is --> be
    car, cars, car's, cars' --> car 

    The result of this mapping of text will be something like:
    the boy's cars are different colors -->the boy car be differ color  
'''

''' 
    Further Explanation with examples at
    http://qr.ae/TUNK22
'''
