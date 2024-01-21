# ashvenk-anikshar-fafaalha-a2

## Part 1

1. **How we formulated the problem:**

    Being a two player game, we proceeded to formulate the solution using *Minimax* with *Alpha-Beta Pruning*. The goal for each player is to make the best move possible in a given state using the most rewarding piece in that state. This can be achieved through Minimax, which is an adversarial search algorithm. To efficiently compute the scores for each move, we also implement Alpha-Beta Pruning.

2. **How the program works:**

    The program begins by converting the 1-D string of the current board state into a 2-D matrix. Then, we call on Minimax to find the best move for the specified player which involves the following steps:
    
    - Choosing a piece: This is based on weights assigned to each piece which helps the program decide which moves for a chosen piece produces the highest score.
    - Calculating scores for each move possible with the chosen piece.
    - Evaluating the moves of each piece and performing Alpha-Beta pruning to remove unrewarding moves.
    - Returning the next board state after performing the best move based on the calculated scores.

    The next board state is converted back from a 2-D matrix to a 1-D string before being returned/printed to stdout.

## Part 2

1. **How we formulated the problem:**

    For this problem, we are mainly working using the *Naive Bayes Classifier* as it showed a lot of great work for problems have similar nature such as the Spam filter based on our research. 

2. **How the program works:**

    The program works by making great use of the trained data as the baseline of what we will consider a Deceptive/Truthful review. It consists of 4 main parts: 
    - **Organizing the Data:** Separate the deceptive reviews and the truthful reviews.
    - **Data Pre-processing:** In this part, we are focusing on cleaning the data itself and making sure we have useful data that we can deal with instead of having the review as a full chunk of text. The pre-processing starts by dividing the review as a list of tokens (words), using the NLTK module. Then we retain the useful tokens only by removing the following from each review:
        - *stopwords* (words that has no meaning such as ‘a’, ‘an’ and etc)
        - *punctuations*
        - *contractions* (e.g. `'ve` from `I've` and `'re` from `you're`)
        - *numerical values*
    - **Applying the Naive Bayes Classifier:** In this part, we calculate the probabilities of each word in the review (from the trained data as well as the test data) that appears for a specific class. Moreover, we take care of edge cases including the case of the word `'x'`, which may not appear in a specific class and will have the probability of 0 were if it is appear in the other class that means the other class will have some thing lower than 0 *(which is a BIG problem)*. 
    - **Labeling the reviews:** in this part we are taking the decision of labeling a review based on the calculation we have applied in the previous part. 

3. **Discussion of problems faced:**

    - The *variation of some words made a lot of noise* for us while cleaning the data, for example `should`, `shouldn't` and `should not` these are three variations where all of them need to be handled and removed.
    - *Building a solid regex was challenging* as we are not allowed to go with the libraries that have built-in functions for extracting the words (such as nltk). So, we needed to decided what to consider word and what is not? should we include the contractions or not? (such as `n't` in `shouldn't`) how to filter out the stopwords? Will numerical values be part of our processing or should we roll them out? (What about years values, don't they make difference?) 
    - *Questions considered when applying the Naive Bayes Classifier:*
        - Should we calculate the probabilities before or after the preprocessing (as the preprocessing stage includes removing many words)?
        - Will this make the calculation different?
        - How do we calculate the Prior Probability? Should we always have equally divided probabilities? For example, if we have two classes we will have 0.5 for each class and if we have three classes we will have 0.33 for each class and so on; or will it be proportionally calculated?
        - Does this mean that the class that has a larger representation in the dataset will have a higher probability?

**Resources:**
- https://dylancastillo.co/nlp-snippets-clean-and-tokenize-text-with-python/#remove-non-alphabetic-characters
- https://docs.python.org/3/library/decimal.html
- https://medium.com/@yashj302/stopwords-nlp-python-4aa57dc492af
- https://en.wikipedia.org/wiki/Naive_Bayes_spam_filtering
- https://www.baeldung.com/cs/naive-bayes-classification-performance
- https://leasetruk.medium.com/naive-bayes-classifier-with-examples-7b541f9ffedf
- https://monkeylearn.com/blog/practical-explanation-naive-bayes-classifier/
