#!/usr/bin/python
#
# Perform optical character recognition, usage:
#     python3 ./image2text.py train-image-file.png train-text.txt test-image-file.png
# 
# Authors: Aniket Sharma, Ashwin Venkatakrishnan and Fares Alharbi
#

from PIL import Image, ImageDraw, ImageFont
import sys
import math

CHARACTER_WIDTH = 14
CHARACTER_HEIGHT = 25

def load_letters(fname):
    im = Image.open(fname)
    px = im.load()
    (x_size, y_size) = im.size
    # print(im.size)
    # print(int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH)
    result = []
    for x_beg in range(0, int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH, CHARACTER_WIDTH):
        result += [ [ "".join([ '*' if px[x, y] < 1 else ' ' for x in range(x_beg, x_beg+CHARACTER_WIDTH) ]) for y in range(0, CHARACTER_HEIGHT) ], ]
    return result

def load_training_letters(fname):
    TRAIN_LETTERS="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    letter_images = load_letters(fname)
    return { TRAIN_LETTERS[i]: letter_images[i] for i in range(0, len(TRAIN_LETTERS) ) }

def getTrainingText(fname):
    data_arr = []
    file = open(fname,'r')
    for hmmSentence in file:
        info = tuple([x for x in hmmSentence.split()])
        data_arr += [(info[0::2], info[1::2]),]
    return data_arr

def getEmissionProbability(test_alphabet, alphabet):
    emission = 0
    for i in range(len(test_alphabet)):
        for j in range(len(test_alphabet[i])):
            try:
                if train_letters[alphabet][i][j] == test_alphabet[i][j] and test_alphabet[i][j] == '*':
                    emission += math.log(0.95)
                elif train_letters[alphabet][i][j] == test_alphabet[i][j] and test_alphabet[i][j] == ' ':
                    emission += math.log(0.65)
                elif train_letters[alphabet][i][j] != test_alphabet[i][j] and test_alphabet[i][j] == ' ':
                    emission += math.log(0.35)
                else:
                    emission += math.log(0.05)
            except KeyError:
                continue
    return emission

def getTransitionProbabilities(trainingData, alpha):
    transitionCount, transitionProb = {}, {}
    for hmmSentence, _ in trainingData:
        for word in hmmSentence:
            prev = " "
            for alphabet in word:
                if alphabet not in transitionCount:
                    transitionCount[alphabet], transitionProb[alphabet] = {}, {}
                    transitionCount[alphabet][prev], transitionProb[alphabet][prev] = 1, 1
                else:
                    if prev not in transitionCount[alphabet]:
                        transitionCount[alphabet][prev] = 1
                    else:
                        transitionCount[alphabet][prev] += 1
                    transitionProb[alphabet][prev] = 1
                prev = alphabet
    for alphabet in transitionCount:
        complete_count = sum(transitionCount[alphabet].values())
        for prev in transitionCount[alphabet]:
            transitionProb[alphabet][prev] = math.log(float(transitionCount[alphabet][prev])/float(complete_count)) * alpha
    return transitionProb

# Simple Bayes Net
def getTextUsingBayesNet(testLetters, alphabets):
    text = ''
    for i in range(len(testLetters)):
        maxEmissionAlphabet = 'A'
        maxEmission = -float('inf')
        for alphabet in alphabets:
            emission = getEmissionProbability(testLetters[i], alphabet)
            if emission > maxEmission:
                maxEmissionAlphabet = alphabet
                maxEmission = emission
        text += maxEmissionAlphabet
    return text

# Hidden Markov Model
def getTextUsingHMM(testLetters, alphabets):
    trainingText = getTrainingText(train_txt_fname)
    hmmMatrix = [[(0,'A') for _ in range(len(alphabets))] for _ in range(len(testLetters))]
    transitionProb = getTransitionProbabilities(trainingText, alpha)
    smoothingFactor = sys.float_info.epsilon
    for i, alphabet in enumerate(alphabets):
        try:
            if alphabet.lower() in transitionProb:
                transition = transitionProb[alphabet.lower()][" "] * alpha
                emission = getEmissionProbability(testLetters[0], alphabet)
                hmmMatrix[0][i] = (transition + emission, alphabet)
            else:
                hmmMatrix[0][i] = (math.log(smoothingFactor)*alpha, alphabet)
        except KeyError:
            continue

    for i in range(1, len(testLetters)):
        for j in range(len(alphabets)):
            maxProbability = -float('inf')
            prev = 'A'
            for k in range(len(alphabets)):
                if alphabets[j].lower() in transitionProb and alphabets[k] in transitionProb[alphabets[j].lower()]:
                    hm = transitionProb[alphabets[j].lower()][alphabets[k]] * alpha
                else:
                    hm = math.log(smoothingFactor) * alpha

                if hmmMatrix[i-1][k][0] + hm > maxProbability:
                    maxProbability = hmmMatrix[i-1][k][0] + hm
                    prev = hmmMatrix[i-1][k][1]

            ans = prev + alphabets[j]
            hmmMatrix[i][j] = (maxProbability + getEmissionProbability(testLetters[i], alphabets[j]), ans)

    max = -float('inf')
    finalSentence = 0
    n = len(testLetters)
    for i in range(len(alphabets)):
        if hmmMatrix[n-1][i][0] > max:
            finalSentence = i

    return hmmMatrix[n-1][finalSentence][1]

################
# Main Program #
################

if len(sys.argv) != 4:
    raise Exception("Usage: python3 ./image2text.py train-image-file.png train-text.txt test-image-file.png")

(train_img_fname, train_txt_fname, test_img_fname) = sys.argv[1:]
train_letters = load_training_letters(train_img_fname)
test_letters = load_letters(test_img_fname)

charset = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
alpha = 1.2

# Get text read by Simple Bayes Net
bayesNetSentence = getTextUsingBayesNet(test_letters, charset)
# Get text read by Hidden Markov Model
hmmSentence = getTextUsingHMM(test_letters, charset)

# The final two lines of your output should look something like this:
# Simple: 1t 1s so orcerec.
#    HMM: It is so ordered.
print("\nSimple: {0}\n   HMM: {1}".format(bayesNetSentence, hmmSentence))