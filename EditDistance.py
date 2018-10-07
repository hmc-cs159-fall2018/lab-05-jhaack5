##
 # Harvey Mudd College, CS159
 # Swarthmore College, CS65
 # Copyright (c) 2018 Harvey Mudd College Computer Science Department, Claremont, CA
 # Copyright (c) 2018 Swarthmore College Computer Science Department, Swarthmore, PA
##

#!/usr/bin/env python3

from numpy import zeros, int8, float64
from collections import defaultdict, Counter
from math import log
import argparse
import pickle
import string
import sys

class EditDistanceFinder():
    DEL, SUB, INS, TRANS = range(4)
    BLANK = '%'
    def __init__(self):
        self.probs = defaultdict(lambda: defaultdict(float))

    def save(self, fp):
        pickle.dump(dict(self.probs), fp)

    def load(self, fp):
        self.probs = defaultdict(lambda: defaultdict(float), pickle.load(fp))

    def _read_misspellings(self, fp):
        misspellings = []
        for line in fp.readlines():
            error, intended = line.split(",")
            misspellings.append((error.strip(), intended.strip()))
        return misspellings        
        
    def train(self, fp):
        misspellings=self._read_misspellings(fp)
        
        last_alignments = None
        alignments = self.train_alignments(misspellings)
        
        while alignments != last_alignments: 
            print("Training...")
            
            last_alignments = alignments
            self.train_costs(alignments)
            alignments = self.train_alignments(misspellings)
            
    def train_alignments(self, misspellings):
        # Align returns distance + alignment.
        # Collapse together the alignment parts only.
        alignments = []
        for observed, intended in misspellings:
            _, this_alignment = self.align(observed, intended)
            alignments.extend(this_alignment)
            
        return alignments
    
    def train_costs(self, alignments):
        counts = defaultdict(Counter) 
        self.probs = defaultdict(lambda: defaultdict(float))

        alphabet = [a for a in string.ascii_lowercase] + ["unk", "%"]

        for a in alphabet:
            for b in alphabet:
                counts[a][b] += .1
        
        for observed_char, intended_char in alignments:
            if len(observed_char) > 1:
                continue # we don't count transpositions
            if observed_char not in alphabet: observed_char = "unk"
            if intended_char not in alphabet: intended_char = "unk"
            counts[intended_char][observed_char] += 1

        for intended_char, counter in counts.items():
            total = sum(counter.values())
            for observed_char, observed_count in counter.items():
                self.probs[intended_char][observed_char] = observed_count / total

    def align(self, observed_word, intended_word):

        table = self._do_align(observed_word, intended_word)
        alignment = self._do_trace(observed_word, intended_word, table)

        return (table['cost'][-1, -1], alignment)

    def _do_align(self, observed_word, intended_word):
        M, N = map(len, (observed_word, intended_word))

        table = zeros((M+1,N+1), dtype=[('cost', float64), ('backtrace', int8)])
    
        for i in range(1,M+1):
            table[i,0] = (table['cost'][i-1,0] + self.ins_cost(observed_word[i-1]), self.INS)
        for j in range(1,N+1):
            table[0,j] = (table['cost'][0,j-1] + self.del_cost(intended_word[j-1]), self.DEL)
    
        for i in range(1,M+1):
            for j in range(1,N+1):
                this_ins = (table['cost'][i-1,j] + self.ins_cost(observed_word[i-1]), self.INS)
                this_del = (table['cost'][i,j-1] + self.del_cost(intended_word[j-1]), self.DEL)
                this_sub = (table['cost'][i-1,j-1] + self.sub_cost(observed_word[i-1], intended_word[j-1]), self.SUB)
                
                # check for a transposition
                this_tra = (float('inf'), self.TRANS)
                if i>1 and j>1 and observed_word[i-1]==intended_word[j-2]      \
                                and observed_word[i-2]==intended_word[j-1]:
                    this_tra = (table['cost'][i-2,j-2] +                       \
                        self.trans_cost(observed_word[i-2],observed_word[i-1]),\
                        self.TRANS)

                table[i,j] = min((this_del, this_sub, this_ins, this_tra))
                
        return table
    
    def _do_trace(self, observed_word, intended_word, table):
        alignments = []
        i, j = map(len, (observed_word, intended_word))
        
        while (i > 0 or j >0):
            this_backtrace = table['backtrace'][i, j]
            if this_backtrace == self.SUB:
                i -= 1
                j -= 1
                alignments.append((observed_word[i],intended_word[j]))
            elif this_backtrace == self.INS:
                i -= 1
                alignments.append((observed_word[i], self.BLANK))
            elif this_backtrace == self.DEL:
                j -= 1
                alignments.append((self.BLANK, intended_word[j]))
            elif this_backtrace == self.TRANS:
                i -= 2
                j -= 2
                alignments.append((observed_word[i:i+2], intended_word[j:j+2]))

        return list(reversed(alignments))
    
    def del_cost(self, char):
        if char not in string.ascii_lowercase:
            char = "unk"
        return 1-self.probs[char][self.BLANK]
    
    def ins_cost(self, char):
        if char not in string.ascii_lowercase:
            char = "unk"
        return 1-self.probs[self.BLANK][char]
    
    def sub_cost(self, observed_char, intended_char):
        if observed_char not in string.ascii_lowercase:
            observed_char = "unk"
        if intended_char not in string.ascii_lowercase:
            intended_char = "unk"
        if observed_char == intended_char: return 0
        else: return 1-self.probs[intended_char][observed_char]

    def trans_cost(self, first_char, second_char):
        return 1 # for simplicity, assume all transpositions are
            # equally likely, and that they are less likely than
            # all other types of errors
        
    def show_alignment(self, alignments):
        observed, intended = list(zip(*alignments))
        print("Observed Word:", " ".join(observed))
        print("Intended Word:", " ".join(intended))

    def pretty_print(self, observed, correct):
        dist, alignments = self.align(observed, correct)
        print("Distance between '{}' and '{}' is {:.5f}".format(observed, correct, dist))
        self.show_alignment(alignments)

    def prob(self, observed_word, intended_word):
        score, alignment = self.align(observed_word, intended_word)
        total_prob = 0
        for observed_char, intended_char in alignment:
            if len(observed_char) > 1:
                total_prob += -3 # assume fixed prob of transposition = 0.001%
                continue
            intd = intended_char if intended_char in self.probs else "unk"
            obsv = observed_char if observed_char in self.probs[intd] else "unk"
            try: 
                total_prob += log(self.probs[intd][obsv])
            except:
                print(observed_word, intended_word)
                sys.exit("Problem with {} and {}".format(intd, obsv))
        return total_prob
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--store", "-s", type=argparse.FileType('wb'), required=True)
    parser.add_argument("--source", type=argparse.FileType('r', encoding='UTF-8'), required=True)
    args = parser.parse_args()

    aligner = EditDistanceFinder()    
    aligner.train(args.source)        
    aligner.save(args.store)
