
## Part 1

1. **How we formulated the problem:**

	We have formulated the problem in the way that we are solving it using the two approaches as requested in the assignment document.
	- The first approach is about considering really simple way to pick what is the most appropriate part-of-speech to be assigned to a specific word, this approach taking only one value in its consideration to make the decision based on which is the the value of emission probability.

	- For the other approach we are considering more advanced procedure that obeying the rules and mechanism of HMM Viterbi algorithm  to make the decision for a specific word what is the best part-of-speech for this exact position in the sentence. We are taking in the consideration not only the emission probability but also two more probabilities which are the transition probability which will tell how good the chance is for the part-of-speech x to be followed by the part-of-speech y. Moreover, we are considering the initial probabilities for the first word of each sentence which will tell how good is the chance of part-of-speech x to be appeared in the beginning in the sentence.

2. **How the program works:**

	Our program works in two different ways:
	- The first method will follow a simple approach to estimate the appropriate part-of-speech for a certain word based only on the emission probability.
	- The second method, HMM using Viterbi, works by following a more complex series of calculations that considers three probabilities in the process of making the taggings which are the initial, emission and transition probabilities. It starts by the initial probability that will find out what is the most suitable part-of-speech for this word to be in the beginning of the sentence, then starting from the second word, our mechanism will use both the emission and the transition probabilities and pick the max value for each tag and taking in the consideration which part-of-speech from the previous word led us to this max value. At the very end when it comes to pick the best part-of-speech for the last word, we are start the backtracing process that will go back following the highest probability of each part-of-speech of each word until the beginning of the sentence. After completing the backtracing process, we are saving them as a list of tags that will have the same length of the original sentence.

3. **Discussion of problems faced, assumptions, simplifications and design decisions:**

	A lot of problems faced during the coding process, especially when it comes to handling the advanced POS tagging system HMM Viterbi. The problems including how should we organize our trained data in order to make it easy to access and make the calculations, It is vage sometimes when it comes between counting or probabilities as we are in need of both of them. Moreover, the too tiny numbers such as x to the power of -23 make it really hard to do a manual debug since these numbers are up to 8 digits. We made the decision to organize our transition probability in a dictionary that look like the following:

		{
			level_0: {
				currentPOS_1: (prevPOS, Prob),
				currentPOS_2: (prevPOS, Prob),
				.
				.
				.
				currentPOS_N: (prevPOS, Prob)
			},
			level_1: {
				currentPOS_1: (prevPOS, Prob), 
				currentPOS_2: (prevPOS, Prob),
				.
				.
				.
				currentPOS_N: (prevPOS, Prob)
			},
			.
			.
			.
			level_N: {
				currentPOS_1: (prevPOS, Prob), 
				currentPOS_2: (prevPOS, Prob),
				.
				.
				.
				currentPOS_N: (prevPOS, Prob)
			}
		}

	where the `level` corresponds to the index of the word in the sentence, for example, *I HAVE A CAR* -> `level_1` is *I*, `level_2` is *HAVE* and so on. And the `currentPOS_N` is one of the list of tags that are assigned to this word, `prevPOS` is the part-of-speech from the previous level that caused the highest probability for `currentPOS_N` and `Prob` is the highest probability calculated for `currentPOS_N`. After all, we are glad that we have HMM with Viterbi working with an accuracy higher than the simplified version which makes sense because HMM with Viterbi requires more factors to be taken into consideration which leads to more accurate POS tagging.

## Part 2

1. **How we formulated the problem:**

	- There are English words and sentences in the picture.
	- The picture uses the same fixed-width, fixed-size typography. For instance, a box of 25 pixels in height and 16 pixels in width encloses each letter.
	- We also assume that we only consider the 26 capital, 26 lowercase, 10 numbers, spaces, and 7 punctuation-symbol symbols found in the Latin alphabet.
	- The skeleton code handles the input/output of the picture, converting it into a list of lists that depicts a two-by-two grid of black and white dots.

2. **How the program works:**

	As with the word-to-word transition probabilities in the previous case, we have found the character-to-character transition probabilities for this one as well Using the emission probability, we may determine which character has the highest likelihood in this manner. After that, we print the string after joining that character to the rest of it.

	- We are keeping the probability values for the dynamic programming technique in a two-column matrix.
	- The ratio of the total number of pixels to the number of matching pixels is how I initially tested it. This was an absolute bust.
	- The next strategy used categorical emission probability based on accuracy level.
	- Additionally, only the emission probabilities were normalised for scaling purposes; the transition probabilities were scaled up somewhat to contribute more to the prediction than the emission probabilities alone due to their extremely low values.

3. **Discussion of problems faced:**

	We struggled to decide which smoothing methods to apply; for example, we experimented with add-k smoothing, Good-Turing smoothing, and Laplace smoothing to handle unobserved transitions in various ways. We observed that the Laplace smoothing approach improved the performance of the code. 

4. **How you divided the work among team members:**

	We have all worked evenly on the tasks.

5. **The contribution of each team member:**

	The team members contributions include doing the starter code, verifying the calculations, debugging and managing the tasks assignments.
