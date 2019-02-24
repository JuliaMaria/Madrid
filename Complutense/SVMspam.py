import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import re
import nltk
import nltk.stem.porter
import os
import io
from sklearn.model_selection import train_test_split

def preProcess(email):

    hdrstart = email.find("\n\n")
    if hdrstart != -1:
        email = email[hdrstart:]

    email = email.lower()
    # Strip html tags. replace with a space
    email = re.sub('<[^<>]+>', ' ', email)
    # Any numbers get replaced with the string 'number'
    email = re.sub('[0-9]+', 'number', email)
    # Anything starting with http or https:// replaced with 'httpaddr'
    email = re.sub('(http|https)://[^\s]*', 'httpaddr', email)
    # Strings with "@" in the middle are considered emails --> 'emailaddr'
    email = re.sub('[^\s]+@[^\s]+', 'emailaddr', email)
    # The '$' sign gets replaced with 'dollar'
    email = re.sub('[$]+', 'dollar', email)
    return email

def email2TokenList(raw_email):
    """
    Function that takes in a raw email, preprocesses it, tokenizes it,
    stems each word, and returns a list of tokens in the e-mail
    """

    stemmer = nltk.stem.porter.PorterStemmer()
    email = preProcess(raw_email)

    # Split the e-mail into individual words (tokens)
    tokens = re.split('[ \@\$\/\#\.\-\:\&\*\+\=\[\]\?\!\(\)\{\}\,\'\"\>\_\<\;\%]',
                      email)

    # Loop over each token and use a stemmer to shorten it
    tokenlist = []
    for token in tokens:

        token = re.sub('[^a-zA-Z0-9]', '', token)
        stemmed = stemmer.stem(token)
        #Throw out empty tokens
        if not len(token):
            continue
        # Store a list of all unique stemmed words
        tokenlist.append(stemmed)

    return tokenlist

def getVocabDict(reverse=False):
    """
    Function to read in the supplied vocab list text file into a dictionary.
    Dictionary key is the stemmed word, value is the index in the text file
    If "reverse", the keys and values are switched.
    """
    vocab_dict = {}
    with open("p6/p6/vocab.txt") as f:
        for line in f:
            (val, key) = line.split()
            if not reverse:
                vocab_dict[key] = int(val)
            else:
                vocab_dict[int(val)] = key

    return vocab_dict

spamFiles = []
for i in os.listdir('p6/p6/spam'):
    spamFile = io.open('p6/p6/spam/' + str(i), mode='r', encoding='utf-8', errors='ignore').read()
    spamFiles.append(email2TokenList(spamFile))

easyHamFiles = []
for i in os.listdir('p6/p6/easy_ham'):
    easyHamFile = io.open('p6/p6/easy_ham/' + str(i), mode='r', encoding='utf-8', errors='ignore').read()
    easyHamFiles.append(email2TokenList(easyHamFile))

hardHamFiles = []
for i in os.listdir('p6/p6/hard_ham'):
    hardHamFile = io.open('p6/p6/hard_ham/' + str(i), mode='r', encoding='utf-8', errors='ignore').read()
    hardHamFiles.append(email2TokenList(hardHamFile))

vocabDict = getVocabDict()

spamFilesEncoded = []
for email in spamFiles:
    encoded = np.zeros((1, 1900))
    for word in email:
        if word in vocabDict:
            encoded[0, vocabDict[word]] = 1
    #Label (spam)
    encoded[0, 0] = 1
    spamFilesEncoded.append(encoded)

easyHamFilesEncoded = []
for email in easyHamFiles:
    encoded = np.zeros((1, 1900))
    for word in email:
        if word in vocabDict:
            encoded[0, vocabDict[word]] = 1
    #Label (non-spam)
    encoded[0, 0] = 0
    easyHamFilesEncoded.append(encoded)

hardHamFilesEncoded = []
for email in hardHamFiles:
    encoded = np.zeros((1, 1900))
    for word in email:
        if word in vocabDict:
            encoded[0, vocabDict[word]] = 1
    #Label (non-spam)
    encoded[0, 0] = 0
    hardHamFilesEncoded.append(encoded)

spamFilesEncoded = np.vstack(spamFilesEncoded)
easyHamFilesEncoded = np.vstack(easyHamFilesEncoded)
hardHamFilesEncoded = np.vstack(hardHamFilesEncoded)

data = np.vstack((spamFilesEncoded, easyHamFilesEncoded, hardHamFilesEncoded))
np.random.shuffle(data)

X = data[:, 1:]
y = data[:, 0]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1)

values = [0.01, 0.1, 1, 10]
for C in values:
    for sigma in values:
        svm = SVC(kernel='rbf', C=C, gamma=float(1) / (2 * sigma ** 2))
        svm.fit(X_train, y_train)
        predict = svm.predict(X_val)
        number = 0
        for i in range(len(y_val)):
            if predict[i] == y_val[i]:
                number = number + 1
        accuracy = (float(number)/len(y_val))*100
        print("Accuracy for C = " + str(C) + " and sigma = " + str(sigma) + " = " + str(accuracy) + "%")
