# This program creates probability counts for each bigram or trigram in the dataset
import json

en_data = open('fr-en/europarl-v7.fr-en.en','r')
lines = en_data.readlines()
en_data.close()
# each dict is an n-gram associated with a count
bigrams = {} 
trigrams = {}
print('starting...')
for line in lines:
    l = line.lower().strip()
    words = l.split()
    # filter out punctuation
    for i in range(len(words)):
        if '.' in words[i] or '?' in words[i] or ',' in words[i]:
            new_term = ''
            for l in words[i]:
                if l != '.' and l != '?' and l != ',':
                    new_term += l
            words[i] = new_term

            
    #print('line:',line)
    #print('words:',words)
    for i in range(1,len(words)):
        hist1 = words[i-1]
        if i > 1:
            hist2 = words[i-2]
            trigram = hist2 + ' ' + hist1 + ' ' + words[i]
            #print('trigram:',trigram)
            if trigram in trigrams:
                trigrams[trigram] += 1
            else:
                trigrams[trigram] = 1

        bigram = hist1 + ' ' + words[i]
        #print('bigram:',bigram)
        if bigram in bigrams:
            bigrams[bigram] += 1
        else:
            bigrams[bigram] = 1




#print("bigram count for 'the people':",bigrams['the people'])
#print("trigram count for 'i should like':",trigrams['i should like'])

with open('trigrams.json','w') as f:
    json.dump(trigrams,f)

with open('bigrams.json','w') as f:
    json.dump(bigrams,f)
