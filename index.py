#https://www.nltk.org/book/ch06.html
import nltk
import random
from nltk.corpus import names



#functions
def gender_features(word):
    return {'name':word}

#Read text files
MaleNames = open("names/male.txt","r")
FemaleNames = open("names/female.txt","r")


#gets labelled names and labels male and female.
labeled_names = ([(name[:-1],'male') for name in MaleNames.readlines()]+[(name[:-1],'female') for name in FemaleNames.readlines()])
random.shuffle(labeled_names)

#returns per feature set a first letter and last letter of a name that has a gender in labeled names
featuresets = [(gender_features(n),gender) for (n,gender) in labeled_names]

#creates a training and testing set
train_set = featuresets[500:]
test_set = featuresets[:1500]

#trains the network.
classifier = nltk.NaiveBayesClassifier.train(train_set)

#tries a name against the network
ToGuess = "Blake"
print("Guessing name: " + ToGuess)
print(ToGuess + " is: " + classifier.classify(gender_features(ToGuess)))

#shows how accurate the classifier is against 1500 names. 
ClassifierAccuracy = nltk.classify.accuracy(classifier,test_set)
FormattedFloat = "{:.2f}".format(ClassifierAccuracy)

print("Accuracy against 1500 names: " + FormattedFloat + "%")
#shows classification examples. 
print(classifier.show_most_informative_features(5))

