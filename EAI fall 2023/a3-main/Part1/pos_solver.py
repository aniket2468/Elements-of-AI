###################################
# CS B551 Fall 2023, Assignment #3
#
# Your names and user ids:
#



import random
import math
import time

# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
class Solver:
    #class values that needed in different locations:
    POS_TAGS = ['start']
    POS_OCCURANCES = [0]
    TrainedDataSentencesCounter = 0
    DEFAULT_TAG = ""
    WORDS_DICTIONARY = {} #dictionary that has the words as keys and nested dictionary for each word that shows how many time this word appears given POS
    #avoid division by zero using the following smoother value
    smootherEpsilon = 1e-12
    #stores tags and a dictionary of what came in front of it along with how many times that happened
    #ex: "noun": {'adv.': 10, 'verb': 55} 
    TAGS_PRECEDENCES_DICTIONARY = {}
    TRANSITION_PROBABILITIES = {}

    
    # Calculate the log of the posterior probability of a given sentence
    # with a given part-of-speech labeling. Right now just returns -999 -- fix this!
    def posterior(self, model, sentence, label): # X: two tags -> which tag will give max posterior
        
        simplePosterior = 0
        HMMPosterior = -999
        if model == "Simple":

            
            for index in range(len(sentence)):
                
                currentWord = sentence[index]
                currentLabel = label[index]
                
                if currentLabel == None:
                    print("none pos detected for word", currentWord, "\nfull sentence", sentence, "\nposes for this word", label)
                    
                simplePosterior += math.log( Solver.emissionOf(currentWord, currentLabel) )
                
            return simplePosterior
        
        elif model == "HMM":
            
            for index in range(len(sentence)):
                
                currentWord = sentence[index]
                currentLabel = label[index]
                
                if currentLabel == None:
                    print("none pos detected for word", currentWord, "\nfull sentence", sentence, "\nposes for this word", label)
                    
                if index == 0:
                    
                    
                    initial = Solver.TAGS_PRECEDENCES_DICTIONARY[label[0]]["start"] / Solver.TrainedDataSentencesCounter
                    emission = Solver.emissionOf(sentence[0], label[0])
                    total = initial * emission
                    HMMPosterior += math.log( total)
                    
                else:
                    
                    transition = Solver.transitionProbabilityCalculator(label[index], label[index-1])
                    emission = Solver.emissionOf(sentence[index], label[index])
                    total = transition * emission
                    
                    
                    HMMPosterior += math.log( total )
                
                
            return HMMPosterior

        else:
            print("Unknown algo!")

            
        
    #helper method -> get the word and returns the emisssion probability for it
    def emissionOf(word, pos):

        wordGivenPOSResult = Solver.smootherEpsilon
        if word in Solver.WORDS_DICTIONARY:
            if pos in Solver.WORDS_DICTIONARY[word]:
                wordGivenPOSResult = Solver.WORDS_DICTIONARY[word][pos]
                
        
        POSCounter = 1
        if pos in Solver.POS_TAGS:
            indexOfPOS = Solver.POS_TAGS.index(pos)
            POSCounter = Solver.POS_OCCURANCES[indexOfPOS]

        emission = wordGivenPOSResult / POSCounter
        return emission

    def makeDummyPreviousDic():
        
        
        
        probabilities = {'adp': 1.72,
                         'start': 2.52,
                         'det': 9.76,
                         'noun': 4.71,
                         'adj': 1.53,
                         'verb': 7.62,
                         '.': 9.73,
                         'adv': 2.50,
                         'conj': 3.54,
                         'prt': 4.98,
                         'pron': 3.28,
                         'num': 8.22,
                         'x': 9.28}
        
        return probabilities
      
    #helper method -> get word and its pos and add it to the WORDS_DICTIONARY to keep words with its counts of each pos
    def addWordToTheDictionary(passedWord, passedPOS):
        
        if passedWord in Solver.WORDS_DICTIONARY:
            
            if passedPOS in Solver.WORDS_DICTIONARY[passedWord]:
                Solver.WORDS_DICTIONARY[passedWord][passedPOS] += 1
            
            else:
                Solver.WORDS_DICTIONARY[passedWord][passedPOS] = 1
            
        else:
            Solver.WORDS_DICTIONARY[passedWord] = {}
            Solver.WORDS_DICTIONARY[passedWord][passedPOS] = 1
            
        

        
    #helper method -> put epsilon value for missing pos
    def fillGapsForMissingPOSForWords():
        
        for word in Solver.WORDS_DICTIONARY:
            for pos in Solver.POS_TAGS:
                
                if pos not in Solver.WORDS_DICTIONARY[word]:
                    Solver.WORDS_DICTIONARY[word][pos] = Solver.smootherEpsilon
                    

        
    #helper method -> takes the tag and what came before the tag and add the tag to the TAGS_PRECEDENCES_DICTIONARY
    def addTagToTheDictionary(passedTag, passedPrecendence)    :
        
        
        if passedTag in Solver.TAGS_PRECEDENCES_DICTIONARY:
            
            if passedPrecendence in Solver.TAGS_PRECEDENCES_DICTIONARY[passedTag]:
                Solver.TAGS_PRECEDENCES_DICTIONARY[passedTag][passedPrecendence] += 1
                
            
            else: 
                Solver.TAGS_PRECEDENCES_DICTIONARY[passedTag][passedPrecendence] = 1
                
                
        else:
            
            Solver.TAGS_PRECEDENCES_DICTIONARY[passedTag] = {}
            Solver.TAGS_PRECEDENCES_DICTIONARY[passedTag][passedPrecendence] = 1
        
    #helper method -> takes the pos and the given pos and return combination string to keep consistency
    def POSGivenPOS(pos, givenPos):
        
        return pos + " given " + givenPos
    
    #helpder method -> calculating the transition probabilities
    def transitionProbabilityCalculator(passedTag, passedPreviousTag):

        #formula used:
            #transition probability = how many times passedTag comes after the passedPreviousTag / number of occurances of passedTag
            #example transition probability of noun | det = how many times the tag noun comes after det / how many times det appears in the data
            

        #transition probability tag1 followed tag2 = tag1 followed tag2 / tag1 
        transitionProbability = 0
        countOfTheTagsAppearedInSameOrder = 0
        
        if passedTag in Solver.TAGS_PRECEDENCES_DICTIONARY:
            if passedPreviousTag in Solver.TAGS_PRECEDENCES_DICTIONARY[passedTag]:
                countOfTheTagsAppearedInSameOrder = Solver.TAGS_PRECEDENCES_DICTIONARY[passedTag][passedPreviousTag]
        
        if passedPreviousTag in Solver.POS_TAGS:
            indexOfPOS = Solver.POS_TAGS.index(passedTag)
            totalCountOfPreviousTag = Solver.POS_OCCURANCES[indexOfPOS]
        
        #avoid division by zero
        transitionProbability = (countOfTheTagsAppearedInSameOrder + 1) / (totalCountOfPreviousTag + len(Solver.POS_TAGS))
        
        return transitionProbability

        

    #helper method -> calculate the transition probabilities of all tags shown in the train data    
    def buildingTransitionProbabilites():
        
        previousTags = ['start', 'det', 'noun', 'adj', 'verb', 'adp', '.', 'adv', 'conj', 'prt', 'pron', 'num', 'x']
        currentTags = ['det', 'noun', 'adj', 'verb', 'adp', '.', 'adv', 'conj', 'prt', 'pron', 'num', 'x']
        
        for preTag in previousTags:
            for currTag in currentTags:
                
                currentKey = Solver.POSGivenPOS(currTag, preTag) 
                Solver.TRANSITION_PROBABILITIES[currentKey] = Solver.transitionProbabilityCalculator(currTag, preTag)

    #helper method -> get word and return the probability of the POSs of this word as it shown in the start of the sentence
    def initialsProbabilitiesOf(word):
        
        #Initial probability = this pos as start divided by number sentences in the trained data
        #multiply the result by the emission to get the total probability
        
        initialProbabilities = {} #{level: {pos1:(start,prob), pos2:(start,prob), ...}}

        for pos in Solver.POS_TAGS:
            
            if pos == "start":
                continue

            currentInitial = Solver.TAGS_PRECEDENCES_DICTIONARY[pos]["start"] / Solver.POS_OCCURANCES[0]
            currentEmission = Solver.emissionOf(word, pos)
            
            initialProbabilities[pos] = ("start", currentInitial * currentEmission)

        return initialProbabilities

    
    #helper method -> gets word and previous probabilites items
    def twoLevelsCalculations(word, previousItemsDic):

        probabilities = {}


        for pos in Solver.POS_TAGS:
            
            if pos == "start":
                continue
            
            #maxTransForCurrentPOS = Solver.smootherEpsilon
            
            prevPOSCauseTheMax = list(previousItemsDic.keys())[0]
            maxTransForCurrentPOS = previousItemsDic[prevPOSCauseTheMax][1]
            
            for prevPos, prevProb in previousItemsDic.items():
                #print("+++ prevPos", prevPos, "+++ prevProb", prevProb)
                
                currentKey = Solver.POSGivenPOS(pos, prevPos) 
                transition = Solver.TRANSITION_PROBABILITIES[currentKey] * prevProb[1]
                #transition = Solver.transitionProbabilityCalculator(pos, prevPos) * prevProb[1]
                #print("transition of key",currentKey, "is ", Solver.TRANSITION_PROBABILITIES[currentKey] ,"maxTransForCurrentPOS", maxTransForCurrentPOS, "while transition", transition)
                if transition > maxTransForCurrentPOS:
                    maxTransForCurrentPOS = transition
                    prevPOSCauseTheMax = prevPos

            probabilities[pos] = (prevPOSCauseTheMax, maxTransForCurrentPOS * Solver.emissionOf(word, pos))

            
        return probabilities

                
        
        
    # Do the training!
    #
    def train(self, data):

        for pair in data:

            words = pair[0]
            POSs = pair[1]
            isItFirstTag = True
            #incremening the 'start' tag manually is we are dealing it as a special case
            Solver.POS_OCCURANCES[0] += 1
            Solver.TrainedDataSentencesCounter +=1
            
            for index in range(len(words)):
                
                currentWord = words[index]
                currentPOS = POSs[index]
                
                if isItFirstTag:
                    Solver.addTagToTheDictionary(currentPOS, "start")
                    isItFirstTag = False
                    
                else:
                    precedenceTag = POSs[index-1]
                    Solver.addTagToTheDictionary(currentPOS, precedenceTag)

                
                Solver.addWordToTheDictionary(currentWord, currentPOS)

                if currentPOS in Solver.POS_TAGS:
                    indexOfPOS = Solver.POS_TAGS.index(currentPOS)
                    
                    #increament the corrosponding counter of POS
                    Solver.POS_OCCURANCES[indexOfPOS] += 1
                    
                else:
                    
                    Solver.POS_TAGS.append(currentPOS)
                    indexOfPOS = Solver.POS_TAGS.index(currentPOS)
                    Solver.POS_OCCURANCES.append(0)

                
        Solver.buildingTransitionProbabilites()
        Solver.fillGapsForMissingPOSForWords()
        Solver.assignTheDefaultTag()


    #helper method -> get word object and returns the pos that has the heighest probability
    def heighestPosProbOf(wordObject):
        
        heighest = wordObject.posProbabilities[0]
        
        if len(wordObject.possiblePOSs) > 1:

            for posProb in wordObject.posProbabilities:
                if posProb > heighest:
                    heighest = posProb
            
            indexOfHeighestPOS = wordObject.posProbabilities.index(heighest)
            heighestPOS = wordObject.possiblePOSs[indexOfHeighestPOS]
            
            
            return (heighest, heighestPOS)
        
        else:
            
            return (heighest, wordObject.possiblePOSs[0])
        
        

    #helper method -> assign the tag with the heighest occurances to the class variable DEFAULT_TAG
    def assignTheDefaultTag():
        tagWithHeighestProbabilityIndex = 0
        hieghest = Solver.POS_OCCURANCES[0]
        
        for tagOccurance in Solver.POS_OCCURANCES:
            
            if tagOccurance > hieghest:
                hieghest = tagOccurance
                
        
        tagWithHeighestProbabilityIndex = Solver.POS_OCCURANCES.index(hieghest)
        defaultTag = Solver.POS_TAGS[tagWithHeighestProbabilityIndex]
        
        Solver.DEFAULT_TAG = defaultTag
            



    # Functions for each algorithm. Right now this just returns nouns -- fix this!
    #
    def simplified(self, sentence):
        
        tags = []
        
        for word in sentence:
            
            if word in Solver.WORDS_DICTIONARY:
                heighestEmission = Solver.smootherEpsilon
                selectedPOS = ""
                
                for pos in Solver.WORDS_DICTIONARY[word]:
                    

                    currentPOSEmission = Solver.emissionOf(word, pos)

                    if  currentPOSEmission > heighestEmission:
                        heighestEmission = currentPOSEmission
                        selectedPOS = pos

                tags.append(selectedPOS)
                        
            else:
                tags.append(Solver.DEFAULT_TAG)
                    
        return tags


    def hmm_viterbi(self, sentence):
        
        
        probability_history_dictionary = {}#{word: {pos1: {prePOS1: prob, prePOS2: prob, prePOS3: prob }, pos2: {pos1: {prePOS1: prob, prePOS2: prob, prePOS3: prob } } } }

        tags = []
        for index, word in enumerate(sentence):

            probability_history_dictionary[index] = {}
            
            if index == 0:
                
                initialProbabilities = Solver.initialsProbabilitiesOf(word)
                #Initial probability = this pos as start divided by number sentences in the trained data
                
                for pos, prob in initialProbabilities.items():

                    probability_history_dictionary[index][pos] = ("start", prob[1]) 
                    
                    #probability_history_dictionary[index][pos] = emission 

            else:
                
                #print("before currentWordProbabilites", word, probability_history_dictionary[index-1])
                currentWordProbabilites = Solver.twoLevelsCalculations(word, probability_history_dictionary[index-1])
                #print("Noe currentWordProbabilites:", currentWordProbabilites)
                probability_history_dictionary[index] = currentWordProbabilites

          
                    
        currentLevel = len(probability_history_dictionary) - 1
        
        
        currentLevelProbs = probability_history_dictionary[currentLevel].items()
        
        currentLevelPOS = list(currentLevelProbs)[0][0]
        highestPrevProb = list(currentLevelProbs)[0][1][1]
        prePOSCauseHighestProb = list(currentLevelProbs)[0][1][0]
            
        for pos, prePos in currentLevelProbs:
            
            if prePos[1] > highestPrevProb:
                currentLevelPOS = pos
                highestPrevProb = prePos[1]
                prePOSCauseHighestProb = prePos[0]
        
        tags.append(currentLevelPOS)
                

        
        currentLevel -= 1
        
        
        while currentLevel >= 0:


            currentLevelProbs = probability_history_dictionary[currentLevel]
            
            if currentLevel == 0:
                
                currentLevelProbs = probability_history_dictionary[currentLevel].items()
                highestYet = list(currentLevelProbs)[0][1][1]
                selectedPOS = list(currentLevelProbs)[0][0]
                

                
                for pos, prob in currentLevelProbs:
                    
                    if prob[1] > highestYet:
                        highestYet = prob[1]
                        selectedPOS =  pos
                        
                
                tags.append(selectedPOS) 
                #print("level zero", probability_history_dictionary[currentLevel].items())
                
            else:    
                
                tags.append(prePOSCauseHighestProb)
                
                prePOSCauseHighestProb = currentLevelProbs[prePOSCauseHighestProb][0]
                highestPrevProb = currentLevelProbs[prePOSCauseHighestProb][1]
            


            currentLevel -= 1

        tags.reverse()
            
        
        return tags
        #return [ "noun" ] * len(sentence)



    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself. 
    # It should return a list of part-of-speech labelings of the sentence, one
    #  part of speech per word.
    #
    def solve(self, model, sentence):
        
        start = time.time()
        
        if model == "Simple":
            return self.simplified(sentence)
            print("program ended with time consumed:", time.time() - start)
        elif model == "HMM":
            return self.hmm_viterbi(sentence)
            print("program ended with time consumed:", time.time() - start)
        else:
            print("Unknown algo!")
            print("program ended with time consumed:", time.time() - start)



