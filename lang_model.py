import json
import random
import math
from itertools import permutations

file = open('trigrams.json','r')
trigrams = json.load(file)
file.close()

# get the total count for all trigrams
#total_count = 0
#for key in trigrams:
#    total_count += trigrams[key]
#print('total_count is',total_count) # this was computed to be 46262909
total_count = 46262909

# extract trigrams from given sentence
def extract_trigrams(s):
    words = s.lower().strip().split()
    trigram_list = list()
    for i in range(len(words)-2):
        trigram_list.append(str(words[i]+' '+words[i+1]+' '+words[i+2]))
    return trigram_list

def prob(t): # calculates the probability of a trigram
    if t in trigrams:
        return trigrams[t] / total_count
    else:
        return 0.5 / total_count # for smoothing
    
def sentence_prob(s): # calculates probability of a sentence
    parts = extract_trigrams(s)
    total_prob = prob(parts[0])
    for i in range(1,len(parts)):
        total_prob *= prob(parts[i])
    return total_prob


def permute_sentence(s):
    sent = s.lower().strip().split()
    random.shuffle(sent)
    while sent == s.lower().strip().split():
        random.shuffle(sent)
    return sent

def recover_sentence(s):
    # generate all combinations of words, rank them probabilistically, return the highest ranked sentence
    s = s.lower().strip()
    parts = s.split()
    permu_list = list(permutations(parts))
    # total possible permutations = n! words
    #total = math.factorial(len(parts))
    #while len(permu_list) < total:
    #    random.shuffle(parts)
    #    if ' '.join(parts) not in permu_list:
    #        permu_list.append(' '.join(parts))
    #print(permu_list)
    # now we have all the possible sentences
    max_prob = 0
    best_sent = ''
    for t in permu_list:
        s = ' '.join(t)
        new_prob = sentence_prob(s)
        #print(s,':',new_prob)
        if new_prob > max_prob:
            best_sent = s
            max_prob = new_prob
            
    return best_sent
        
        
        

## to normalize probabilities, divide by length of dictionary
# so, comparing one sentence with another is about comparing final prob score

#sentence = 'I declare resumed the session of the European Parliament adjourned'
#sentence = "Please rise, then, for this minute' s silence."
#sentence = 'I like cats because they are cuddly and cute'
#permuted_sent = ' '.join(permute_sentence(sentence))
#print(permuted_sent)
#print(recover_sentence(permuted_sent))
#print(extract_trigrams(sentence))
#mixed_sentence = # every combo of words, score according to 3-gram model
# 3 gram model of sentence would be:
# p(cute, cuddly and) * p(and, are cuddly) * p(cuddly, they are) * ... * p(cats, I like)
# = p(cuddly and cute) * p(are cuddly and) * p(they are cuddly) * ... * p(I like cats)
# p(w3, w1 w2) = p(w1 w2 w3) = count(w1 w2 w3)/(all trigram counts)

### RECONSTRUCTION TESTING
with open('fr-en/europarl-v7.fr-en.en','r') as f:
    en_lines = f.readlines()
perfect = 0
prev_used = set()
x = 0
for i in range(len(en_lines)):
    if x >= 38:
        break
    if len(en_lines[i].split()) < 11 and len(en_lines[i].split()) > 3 and en_lines[i].lower().strip() not in prev_used:
        prev_used.add(en_lines[i].lower().strip())
        print('original:',en_lines[i].lower().strip())
        permuted_sent = ' '.join(permute_sentence(en_lines[i]))
        recovered = recover_sentence(permuted_sent).strip()
        print('recovered:',recovered)
        if en_lines[i].lower().strip() == recovered:
            print('perfect!')
            perfect += 1
        x += 1
        print('\n')

print(perfect,'recovered exactly')
    

