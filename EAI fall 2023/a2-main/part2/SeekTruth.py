# SeekTruth.py: Classify text objects into two categories
#
# Fares Alharbi: fafaalha
# Ashwin Venkatakrishnan: ashvenk
# Aniket Sharma: anikshar

import sys
import re
import time
from decimal import Decimal
from tqdm import tqdm

def load_file(filename):
    objects=[]
    labels=[]
    with open(filename, "r") as f:
        for line in f:
            parsed = line.strip().split(' ',1)
            labels.append(parsed[0] if len(parsed) > 0 else "")
            objects.append(parsed[1] if len(parsed) > 1 else "")
    return {"objects": objects, "labels": labels, "classes": list(set(labels))}

# classifier: Train and apply a bayes net classifier
#
# This function should take a train_data dictionary that has three entries:
#        train_data["objects"] is a list of strings corresponding to reviews
#        train_data["labels"] is a list of strings corresponding to ground truth labels for each review
#        train_data["classes"] is the list of possible class names (always two)
#
# and a test_data dictionary that has objects and classes entries in the same format as above. It
# should return a list of the same length as test_data["objects"], where the i-th element of the result
# list is the estimated classlabel for test_data["objects"][i]
#
# Do not change the return type or parameters of this function!
#
def classifier_original(train_data, test_data):
    return [test_data["classes"][0]] * len(test_data["objects"])


def classifier(train_data, test_data):
    train_data_deceptiveReviews = getReviewsOf('deceptive', train_data)
    train_data_truthfulReviews = getReviewsOf('truthful', train_data)
    train_data_deceptiveWords = getWordsOf(train_data, 'deceptive')
    train_data_truthflWords = getWordsOf(train_data, 'truthful')
    # calculating the prior probability assuming that the review is deceptive initially
    priorProbabilityOfDeceptive = Decimal(len(train_data_deceptiveReviews) / (len(train_data_deceptiveReviews) + len(train_data_truthfulReviews)))
    # calculating the prior probability assuming that the review is truthful initially
    priorProbabilityOfTruthful = Decimal(len(train_data_truthfulReviews) / (len(train_data_deceptiveReviews) + len(train_data_truthfulReviews)))
    test_reviews = test_data['objects']
    potentialLabels = []
    counter = 0
    for review in tqdm(test_reviews, desc="Classifying Data", ncols=100):
        (_, potentialLabel) = getThePotentialLabelFor(review, train_data_deceptiveWords, train_data_truthflWords, priorProbabilityOfDeceptive, priorProbabilityOfTruthful)
        potentialLabels.append(potentialLabel)
        counter += 1
    return potentialLabels

def getReviewsOf(label, inData):
    reviews = []
    for index in range(len(inData['objects'])):
        if inData['labels'][index] == label:
            reviews.append(inData['objects'][index])
    return reviews

# helper method -> returning the cumulative words (including repetitions) showed in the 'data' for exact 'label' value
def getWordsOf(data, label):
    words = []
    for index in range(len(data['objects'])):
        if data['labels'][index] == label:
            review = data['objects'][index]
            # cleaning the review from punctuation which will help us in the upcoming stepss
            reviewTokens = tokensOf(review)
            cleanedReview = clean(reviewTokens)
            for word in cleanedReview:
                words.append( word )
    return words

# helper method to find the if the 'word' is a stoop word or not
def is_stopword(word):   
    # these are the stopwords from SpyCy library, I'm having them here in order to reduce the elapsed time needed to run the code
    stopwords = ['call', 'upon', 'still', 'nevertheless', 'down', 'every', 'forty', '‘re', 'always',
    'whole', 'side', "n't", 'now', 'however', 'an', 'show', 'least', 'give', 'below', 'did',
    'sometimes', 'which', "'s", 'nowhere', 'per', 'hereupon', 'yours', 'she', 'moreover',
    'eight', 'somewhere', 'within', 'whereby', 'few', 'has', 'so', 'have', 'for', 'noone',
    'top', 'were', 'those', 'thence', 'eleven', 'after', 'no', '’ll', 'others', 'ourselves',
    'themselves', 'though', 'that', 'nor', 'just', '’s', 'before', 'had', 'toward', 'another',
    'should', 'herself', 'and', 'these', 'such', 'elsewhere', 'further', 'next', 'indeed', 'bottom',
    'anyone', 'his', 'each', 'then', 'both', 'became', 'third', 'whom', '‘ve', 'mine', 'take', 'many',
    'anywhere', 'to', 'well', 'thereafter', 'besides', 'almost', 'front', 'fifteen', 'towards', 'none',
    'be', 'herein', 'two', 'using', 'whatever', 'please', 'perhaps', 'full', 'ca', 'we', 'latterly',
    'here', 'therefore', 'us', 'how', 'was', 'made', 'the', 'or', 'may', '’re', 'namely', "'ve", 'anyway',
    'amongst', 'used', 'ever', 'of', 'there', 'than', 'why', 'really', 'whither', 'in', 'only', 'wherein',
    'last', 'under', 'own', 'therein', 'go', 'seems', '‘m', 'wherever', 'either', 'someone', 'up', 'doing',
    'on', 'rather', 'ours', 'again', 'same', 'over', '‘s', 'latter', 'during', 'done', "'re", 'put', "'m",
    'much', 'neither', 'among', 'seemed', 'into', 'once', 'my', 'otherwise', 'part', 'everywhere', 'never',
    'myself', 'must', 'will', 'am', 'can', 'else', 'although', 'as', 'beyond', 'are', 'too', 'becomes',
    'does', 'a', 'everyone', 'but', 'some', 'regarding', '‘ll', 'against', 'throughout', 'yourselves', 'him',
    "'d", 'it', 'himself', 'whether', 'move', '’m', 'hereafter', 're', 'while', 'whoever', 'your', 'first',
    'amount', 'twelve', 'serious', 'other', 'any', 'off', 'seeming', 'four', 'itself', 'nothing', 'beforehand',
    'make', 'out', 'very', 'already', 'various', 'until', 'hers', 'they', 'not', 'them', 'where', 'would', 'since',
    'everything', 'at', 'together', 'yet', 'more', 'six', 'back', 'with', 'thereupon', 'becoming', 'around',
    'due', 'keep', 'somehow', 'n‘t', 'across', 'all', 'when', 'i', 'empty', 'nine', 'five', 'get', 'see', 'been',
    'name', 'between', 'hence', 'ten', 'several', 'from', 'whereupon', 'through', 'hereby', "'ll", 'alone', 'something',
    'formerly', 'without', 'above', 'onto', 'except', 'enough', 'become', 'behind', '’d', 'its', 'most', 'n’t',
    'might', 'whereas', 'anything', 'if', 'her', 'via', 'fifty', 'is', 'thereby', 'twenty', 'often', 'whereafter',
    'their', 'also', 'anyhow', 'cannot', 'our', 'could', 'because', 'who', 'beside', 'by', 'whence', 'being', 'meanwhile',
    'this', 'afterwards', 'whenever', 'mostly', 'what', 'one', 'nobody', 'seem', 'less', 'do', '‘d', 'say', 'thus', 'unless',
    'along', 'yourself', 'former', 'thru', 'he', 'hundred', 'three', 'sixty', 'me', 'sometime', 'whose', 'you', 'quite', '’ve',
    'about', 'even', 'the','and','in', 'is', 'it' , 'of', 'for', 'this', 'a', 'an', 'i', 'me', 'my', 'myself', 'we',
    'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves',
    'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
    'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was',
    'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the',
    'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
    'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in',
    'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
    'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
    'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm',
    'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn',
    'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn']
    
    return True if word in stopwords else False
    
    
    
#helper method -> takes review as a parameter, returns a list of words in the review
def tokensOf(review):
    words = re.findall(r'\b[\w\']+\b', review)
    return clean(words)

    
# helper method -> takes a review as a parameter, returns the review without stop words and punctuations
def clean(review):
    cleanedReview = []
    for word in review:
        word = word.lower()
        # removing stop words
        if is_stopword(word) == False:
            # removing punctuations
            word = re.sub(r'[^\w\s]','', word)
            # removing possession s 
            word = re.sub(r"'s\b", "", word)
            # removing the contraction not such as shouldn't couldn't
            word = re.sub(r"n't\b", "", word)
            # removing the contraction have such as I've
            word = re.sub(r"'ve\b", "", word)
            # removing the contraction are such as we're
            word = re.sub(r"'re\b", "", word)
            if len(word) < 3:
                continue
            # making sure it is not numerical value
            if word.isalpha():
                cleanedReview.append(word.lower())
    return cleanedReview
    
    
# helper method -> count number of the 'word' occurrences in the 'inData'
def occurrencesOf(word, inData):
    occurrencesList = []
    for cWord in inData:
        if cWord == word:
            occurrencesList.append(cWord)
    return len(occurrencesList)

# helper method -> calculate the probability of the appearance of the 'word' in 'inData'
def probabilityOfAppearance(word, inData):
    # adding one to avoid ending up with overall value of 0 in case of probability of 0 for any component (alpha)
    wordOccurrencesInData = occurrencesOf(word, inData) + 1
    totalWordsCountInData = len(inData)
    return wordOccurrencesInData / totalWordsCountInData



def getThePotentialLabelFor(review, deceptiveWords, truthfulWords, deceptivePriorProb, truthfulPriorProb):
    priorPobabilityOfBeingDeceptive = deceptivePriorProb
    priorProbabilityOfBeingTruthful = truthfulPriorProb
    reviewTokens = tokensOf(review)

    for word in reviewTokens:
        # get the probability for the word to appear in a deceptive review
        probToAppeareInDeceptive = probabilityOfAppearance(word, deceptiveWords)
        # get the probability for the word to appear in a truthful review
        probToAppeareInTruthful = probabilityOfAppearance(word, truthfulWords) 
        priorPobabilityOfBeingDeceptive *= Decimal(probToAppeareInDeceptive)
        priorProbabilityOfBeingTruthful *= Decimal(probToAppeareInTruthful)
    
    if priorPobabilityOfBeingDeceptive > priorProbabilityOfBeingTruthful:
        return (priorPobabilityOfBeingDeceptive, 'deceptive')
    else: 
        return (priorProbabilityOfBeingTruthful, 'truthful')



def getThePotentialLabelFor_removingWordsWithMeanValue(review, passedDeceptiveData, passedTruthfulData):
    # calculating the prior probability assuming that the review is deceptive initially
    deceptiveLabelPriorProbability = len(passedDeceptiveData) / (len(passedDeceptiveData) + len(passedTruthfulData))
    # calculating the prior probability assuming that the review is truthful initially
    truthfulLabelPriorProbability = len(passedTruthfulData) / (len(passedDeceptiveData) + len(passedTruthfulData))
    # get the number of words for each label
    train_data_deceptiveReviewsWords = getWordsOf(train_data, 'deceptive')
    train_data_truthfulReviewsWords = getWordsOf(train_data, 'truthful')
    reviewTokens = tokensOf(review)
    for word in reviewTokens:
        probToAppeareInDeceptive = probabilityOfAppearance(word, train_data_deceptiveReviewsWords)
        if probToAppeareInDeceptive < 0.45 or probToAppeareInDeceptive > 0.55:
            # adding one to avoid ending up with overall value of 0 in case of probability of 0 for any component (alpha)
            deceptiveLabelPriorProbability *= ( probabilityOfAppearance(word, train_data_deceptiveReviewsWords) + 1)
            
        # ignore those words where its probability to appear in truthful review is almost %50
        probToAppeareInTruthful = probabilityOfAppearance(word, train_data_truthfulReviewsWords)
        if probToAppeareInTruthful < 0.45 and probToAppeareInTruthful > 0.55:
            # adding one to avoid ending up with overall value of 0 in case of probability of 0 for any component (alpha)
            truthfulLabelPriorProbability *= ( probabilityOfAppearance(word, train_data_truthfulReviewsWords) + 1)
    
    if deceptiveLabelPriorProbability > truthfulLabelPriorProbability:
        print("result:", deceptiveLabelPriorProbability, truthfulLabelPriorProbability, " -> deceptive", reviewTokens[:4])
        return (deceptiveLabelPriorProbability, 'deceptive')
    else: 
        print("result:", deceptiveLabelPriorProbability, truthfulLabelPriorProbability, " -> truthful", reviewTokens[:4])
        return (truthfulLabelPriorProbability, 'truthful')

if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise Exception("Usage: classify.py train_file.txt test_file.txt")

    (_, train_file, test_file) = sys.argv
    # Load in the training and test datasets. The file format is simple: one object
    # per line, the first word one the line is the label.
    train_data = load_file(train_file)
    test_data = load_file(test_file)
    if(sorted(train_data["classes"]) != sorted(test_data["classes"]) or len(test_data["classes"]) != 2):
        raise Exception("Number of classes should be 2, and must be the same in test and training data")

    # make a copy of the test data without the correct labels, so the classifier can't cheat!
    test_data_sanitized = {"objects": test_data["objects"], "classes": test_data["classes"]}

    start = time.time()
    results = classifier(train_data, test_data_sanitized)

    # calculate accuracy
    correct_ct = sum([(results[i] == test_data["labels"][i]) for i in range(0, len(test_data["labels"]))])
    print("Classification accuracy = %5.2f%%" % (100.0 * correct_ct / len(test_data["labels"])))
    print("Time consumed:", time.time() - start)
