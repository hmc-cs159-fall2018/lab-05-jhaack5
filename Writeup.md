# Lab 5

Author: Jordan Haack

## Part 1

1) Laplace smoothing is the procedure for adding some default non-zero amount to each possible count. This is identical to a weighted combination of a uniform distribution and the observed distribution. In practice, it takes probability mass from the observed events with high costs, and adds it to the probability mass of observed events with a low cost. In the EditDistance.py file, Laplace smoothing is implemented by adding .1 to the counts of every possible observed/intended char pairs (including blank/unknown chars). This has the added effect of making all of these counts positive and non-zero. This fixes the problem in the prob method, which would assign the entire observed/intended word pair a probabilility of zero if any edits that don't appear in the training set appear in this word pair. Note, this situation is possible since minimizing edit distance cost isn't the same as maximizing probaility with this definition, because we cap the cost of any edit at 1, even if it doesn't appear in the training set. Finally, if we leave in zero probabilities, we will run into negative infinity when we take the logarithm of it. So, the Laplace smoothing also adds a lower bound to the probability for any unobserved character error.

2) The command-line interface for EditDistance.py takes two required inputs. The first input 'store' represents the location of the file that we write the probability matrix to. The program uses pickle to store this data. The second input 'source' represents the location of the file that we will read input from to train our probabilities. An example way to run this is: python3 EditDistance.py --store ed.pkl --source /data/spelling/wikipedia_misspellings.txt. 

## Part 2

3) The unigram and bigram orders are supported by the given LanguageModel class.

4) The LanguageModel class deals with zero counts using laplace smoothing. The amount of smoothing is determined by the input parameter alpha, which defaults to 0.1.

5) The vocabulary of the language model is the max_vocab (default 40000) most common words in the training data, plus 3 special words for start/end of sentence and unknown words. The \_\_contains\_\_ method returns whether the input word is in the vocabulary of the LanguageModel instance.

6) The get_chunks method is a generator. So, when the get_chunks method is called, it returns a generator which will generate "chunks" of data from the list of source files, generating one at a time to save memory. Each time the generator is asked for another chunk, the generator will read the next few lines of the current source file (up to chunk\_size bytes). If it does this, it will return those lines, using \\n as a separator between the lines. If it hits the end of a file, it will move on to the next source file. Once it has exhaused the source files, it will be done generating. This construction prevents all of the text from all of the source files from being loaded into memory at once.

7) The command-line interface for LanguageModel.py takes two required inputs. The first input 'store' represents the location of the file that we will use to write our LanguageModel instance (again, pickle is used for this). The second required input is the 'source', which is a list of files (where we can also use regular expressions to list more than one file) which will be used for training our vocabulary and unigram/bigram models. There are two optional parameters. Alpha (default 0.1) is how much Laplace smoothing to do. Vocab (default 40000) is the maximum size of the vocab we will allow. We choose this many of the most common words in our training set to be our vocab, and consider everything else unknown. An example usage is: python3 LanguageModel.py --store lm.pkl /data/gutenberg/*.txt

## Part 3

6) One drawback of our SpellChecker is its limited dictionary. For example, the word 'edit' was not found in the dictionary, so we changed it to 'exit'. Another example is that ispell takes 'foetus' to 'fetus' while our spell checker doesn't correct that word. More training data would help solve this issue. Another drawback of our spellchecker is that it isn't build to handle capitalizations or punctuation. I made a slight improvement to uor spell checker by lowercasing everything, and by not spell checking common punctuation marks. Our spell checker also fails at transoposition errors. For example, ispell has 'chirst' to 'christ' while our spell checker says 'thirst', even following the word 'jesus'. Our spell checker only tries to insert letters and never spaces (since insert and substitute only return lists of single words), so we miss errors such as 'thankyou'. To be honest I couldn't find very many examples of when our spell checker does better than ispell. One example is that ispell sometimes makes unnecessary corrections to names, such as 'gustave' to 'gust ave', and our spell checker does this less often. You could probably cook up some examples where our spell ckecker uses the context to correctly replace a misspelled word, but I didn't see any in the reddit comments. 

7) Ispell seems to be best at detecting errors that are one substitution, deletion, insertion, or transposition away from the error. Our spell checker seems to be best at detecting errors that are one/two substitutions, deletions, or insertions away, and/or if the replacement word is a common word (i.e. with high unigram/bigram score), assuming the word appeared in our training data. 

8) I may have made some sort of mistake in training our model, but it seems to me that our spell checker makes more mistakes than corrections. As in, the comments looked better before running the spell checker. The same might even be true for ispell as well. When words are only 1 edit distance away, it is usually very easy to see what the intended word was. But, if the spell checker replaces it with the wrong word, it becomes much harder to determine the writer's intent. I also want to note that our spell checker doesn't check every word that is edit distance 2 away from the misspelling, because it only checks words that are 1 edit distance away from an actual word that is itself 1 edit distance away. I.e. if we type 'helli', we will consider 'bell' as a substitution because 'hell' is a word. But if we type 'jelli' we will not consider 'bell' as a replacement.

## Part 4 - Transpositions

9) My approach to handle transpositions lies mostly in the EditDistance.py file. In the dynamic programming algorithm for edit distance, I add a fourth rule. If both the current part of the intended/observed words are at least 2 letters, and the last two letters of each match a transposition, I allow a transposition to occur. I also modified the alignment and backtracers to handle transpositions. Finally, I added a fourth type of single edit word generator in SpellCheck.py, so that we now consider candidate words due to transpositions.

10) This approach clearly has some merit. For example, the transposition is the lowerst cost way to align 'helol' and 'hello', taking the cost from 1.8 to 1, and taking the log probability from -5.51 to -3.46. This is good, because we expect this to be a transposition error. I also tested this method on the sample sentence 'they did not yb any menas'. Now, 'means' is the top suggestion for the final word, instead of 'men', showing another improvement. Finally, I also tested the sentence 'jesus chirst look at that', and now 'christ' is the top suggestion instead of 'thirst'. Overal, it appears that this approach is better than the baseline model.

11) To avoid complications in training, I made transpositions have a fixed cost of 1, which makes them more costly than any other type of error, but cheaper than the current cost of a transposition, which is either two substitutions or an insertion+deletion. I also gave transpositions a probability of 0.1% of occuring when computing probability (this probability is on the order of the probability for other types of errors). Again, this avoids having to train counts on the transpositions as well. I think it is reasonable to assume that most transpositions occur with roughly the same cost/probability when typing, though this method could be improved by counting transpoisition errors in training data. You would define the probability of an 'ab' transposition to be the count of 'ba' misspellings divided by the count of 'ab' in the intended words. You might find that transpositions are even more likely than other types of errors. The constant cost method I used is similar to just doing a lot of smoothing on the transpositions. 
