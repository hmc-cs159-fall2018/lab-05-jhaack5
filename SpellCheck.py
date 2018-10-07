# Author: Jordan Haack
# CS159 Lab 5
# This file implements a spell checker

from LanguageModel import LanguageModel
from EditDistance import EditDistanceFinder
import spacy
import string

class SpellChecker:

    def __init__(self, max_distance, channel_model=None, language_model=None):
        self.nlp = spacy.load("en", pipeline=["tagger", "parser"])
        self.channel_model = channel_model
        self.language_model = language_model
        self.max_distance = max_distance
        self.punc = '.?:;"\'!\n,/\\'

    def load_channel_model(self, fp):
        self.channel_model = EditDistanceFinder()
        self.channel_model.load(fp)

    def load_language_model(self, fp):
        self.language_model = LanguageModel()
        self.language_model.load(fp)

    def bigram_score(self, prev_word, focus_word, next_word):
        # returns the average log prob between bigrams (prev, focus) 
        # and (focus, next).
        return 0.5 * (self.language_model.bigram_prob(prev_word, focus_word)
                     + self.language_model.bigram_prob(focus_word, next_word) )

    def unigram_score(self, word):
        # returns the log probability of this unigram
        return self.language_model.unigram_prob(word)

    def cm_score(self, error_word, corrected_word):
        # returns the log probability that error_word was typed when
        # corrected_word was intended.
        return self.channel_model.prob(error_word, corrected_word)

    def inserts(self, word):
        # returns a list of words that are within one insert of word
        # we try inserting each character in each position of the word
        l = []
        for i in range(len(word) + 1):
            for new_char in string.ascii_lowercase:
                new_word = word[0:i] + new_char + word[i:]
                if new_word in self.language_model:
                    l.append(new_word)
        return list(set(l))

    def deletes(self, word):
        # returns a list of words that are within one delete of word
        # we try deleting each character in the word
        l = []
        for i in range(len(word)):
            new_word = word[0:i] + word[i+1:]
            if new_word in self.language_model:
                l.append(new_word)
        return list(set(l))

    def substitutions(self, word):
        # returns a list of words that are within one substitution of word
        # we try substituting each character in the word with every other char
        l = []
        for i,char in enumerate(word):
            for new_char in string.ascii_lowercase:
                new_word = word[0:i] + new_char + word[i+1:]
                if char != new_char and new_word in self.language_model:
                    l.append(new_word)
        return l

    def transpositions(self, word):
        # returns a list of words that are within one transposition of word
        # we try transposing each pair of adjacent characters
        l = []
        for i in range(len(word)-1):
            new_word = word[0:i] + word[i+1] + word[i] + word[i+2:]
            if new_word in self.language_model:
                l.append(new_word)
        return list(set(l))

    def generate_candidtates(self, word):
        # returns a list of words that are within max_distance
        # edits from the input word. We do this by first generating
        # the words that are 1 edit away, then the words that are 1
        # edit away from those, and so on.

        checked_word_list = []   # tracks words we've already checked
        words_to_check = [word]  # tracks words we need to check
        word_list = []           # tracks our final word list

        for _ in range(self.max_distance):

            new_words_list = []  # new words on this iteration

            for w in words_to_check:
                if w in checked_word_list:
                    continue     # we alreadly checked this word
                checked_word_list.append(w)
                
                # try deletion/insertion/substitution to find new words
                new_words_list.extend(self.inserts(w))
                new_words_list.extend(self.deletes(w))
                new_words_list.extend(self.substitutions(w))
                new_words_list.extend(self.transpositions(w))

            # add new unique words to our word list
            words_to_check = []
            for w in new_words_list:
                if w not in word_list:
                    word_list.append(w)
                    words_to_check.append(w)

        return word_list

    def sort_candidates(self, error_word, prev_word, next_word, candidates):
        """ takes as input a spelling error and a list of candidates
            and returns a sorted list of candidates, where earlier candidates
            are "better" suggestions, in terms of a weighted combination of
            unigram score, bigram score, and edit distance score.
            Note, our choice depends somewhat on the context of the word
        """
        score_list = []
        for candidate in candidates:
            bigram_score = self.bigram_score(prev_word, candidate, next_word)
            unigram_score = self.unigram_score(candidate)
            edit_score = self.cm_score(error_word, candidate)
            # we use an equally weighted linear combination of log edit score
            # and language model score.
            score = 0.5 * edit_score + 0.25 * (bigram_score + unigram_score)
            score_list.append((candidate, score))

        # sort list so that highest score comes first
        sorted_list = sorted(score_list, key=lambda x: -x[1])
        return [w for w,s in sorted_list]

    def check_non_words(self, sentence, fallback=False):
        """ Takes as input a list of words, and returns a list of lists
            of words. If the word is in the language model, the list
            contains just the original word. Otherwise, it contains a list
            of spell correcting suggestions. If fallback is true, we will 
            replace any word with no suggestions with the list of just
            the word itself.
        """
        l = []

        for i,word in enumerate(sentence):
            word = word.lower() # enfore lowercase
            if word in self.language_model or word in self.punc:
                l.append([word]) # correctly spelled word/punctuation
            else:
                candidates = self.generate_candidtates(word)
                prevW = sentence[i-1] if i>0 else "<s>"
                nextW = sentence[i+1] if i+1<len(sentence) else "</s>"
                canditates = self.sort_candidates(word,prevW,nextW, candidates)

                if canditates or not fallback:
                    l.append(canditates) # give candidate suggestions
                else:
                    l.append([word]) # fallback case, no candidates

        return l
                

    def check_sentence(self, sentence, fallback=False):
        """ Takes as input a list of words, and returns a list of 
            lists of words. Correctly spelled words appear in their own
            list; otherwise, a list of spelling corrections is given in
            order of likelihood.
        """
        return self.check_non_words(sentence, fallback=fallback)

    def check_line(self, line, fallback=False):
        """ Takes as input a string, tokenizes it, and returns a list of 
            lists of words. Correctly spelled words appear in their own
            list; otherwise, a list of spelling corrections is given in
            order of likelihood.
        """
        doc = self.nlp(line) # use spacy to segment sentences
        l = []

        for sent in doc.sents:
            # genreate sentence as list of strings
            # ignore punctuation characters
            sentence = [str(w) for w in sent]
            # pass our sentence to the check_sentence method
            l.extend(self.check_sentence(sentence, fallback=fallback))

        return l
    
    def autocorrect_sentence(self, sentence):
        """ takes a list of tokens and returns a new list of tokens 
            where each non-word has been replaced by its most likely 
            spelling correction.
        """
        l = self.check_sentence(sentence, fallback=True)
        return [w[0] for w in l]

    def autocorrect_line(self, line):
        """ takes a string as input, tokenizes and segment it with spacy, 
            and then returns the concatenation of the result of calling 
            autocorrect_sentence on all of the resulting sentence objects
        """
        doc = self.nlp(line) # use spacy to segment sentences
        l = []

        for sent in doc.sents:
            sentence = [str(w) for w in sent]
            l.extend(self.autocorrect_sentence(sentence))

        return l

    def suggest_sentence(self, sentence, max_suggestions):
        """ Takes as input a list of words, and returns a list of 
            words and lists of words. Correctly spelled words appear on their
            own; otherwise, a list of spelling suggestions is given in
            order of likelihood.
        """
        suggestions = self.check_sentence(sentence, fallback=True)
        l = []

        for i in range(len(sentence)):
            if sentence[i] in self.language_model or sentence[i] in self.punc:
                l.append(sentence[i])
            else:
                l.append(suggestions[i][0:max_suggestions] )
        return l

    def suggest_line(self, line, max_suggestions):
        """ takes a string as input, tokenizes and segments it with spacy, 
            and then returns the concatenation of the result of calling 
            suggest_sentence on all of the resulting sentence objects
        """
        doc = self.nlp(line) # use spacy to segment sentences
        l = []

        for sent in doc.sents:
            sentence = [str(w) for w in sent]
            l.extend(self.suggest_sentence(sentence, max_suggestions))

        return l