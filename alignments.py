## Note that this work will be ENGLISH --> French

from itertools import permutations
#import json
import pickle

# generate all possible alignments given the length of french and english sentences
def get_alignments(fr_len, en_len):
    # simple cases
    if en_len == 1:
        x = [[set(i for i in range(1,fr_len+1))]]
        x[0] = tuple(x[0])
        return x
    
    alignments = list()
    fr = range(1,fr_len+1)
    fr = list(fr) + [0]*(en_len-fr_len)        
    # get all possible orderings
    fr_permu = permutations(fr)
 
    for permu in fr_permu:
        #print(permu)
        numbers = list(permu) #[1,2,3]
        #print(len(numbers),'numbers')
        for i in range(len(numbers)):
            for j in range(i,len(numbers)):
                new_align = numbers.copy()
                if numbers[i] != 0:
                    new_align[i] = [new_align[i]]
                    if i <= len(numbers)-2:
                        for k in range(i+1,j+1):
                            if numbers[k] != 0:
                                new_align[i].append(numbers[k])
                                new_align[k] = 0
                        if len(new_align[i]) > 1:
                            new_align[i] = set(new_align[i])
                        else:
                            new_align[i] = new_align[i][0]
                        for p in permutations(new_align):
                            if p not in alignments:
                                alignments.append(p)


    '''
    # delete all alignments with words having fertility > 25
    i = 0
    while i < len(alignments):
        s = False
        align = alignments[i]
        for j in align:
            if type(j) == set and len(j) > 25:
                s = True
        if s:
            alignments.pop(i)
        else:
            i += 1
    '''

            
    if en_len < fr_len:
        # need to delete a lot of stuff
        i = 0
        while i < len(alignments):
            s = False
            el = 0
            while el < len(alignments[i]):
                if type(alignments[i][el]) == set:
                    s = True
                    break
                '''
                elif alignments[i][el] == 0:
                    alignments[i] = list(alignments[i])
                    print('removing',alignments[i][el],'from',alignments[i])
                    alignments[i].pop(el)
                    el -= 1
                    alignments[i] = tuple(alignments[i])
                    print(alignments[i])
                '''
                el += 1
            if not s:
                #print('removing', alignments[i])
                alignments.remove(alignments[i])
                i -= 1
            i += 1
        
        i = 0
        
        while i < len(alignments):
            '''
            if len(alignments[i]) > en_len:
                alignments.pop(i)
            '''
            while len(alignments[i]) > en_len:
                alignments[i] = list(alignments[i])
                #print(alignments[i])
                if 0 in alignments[i]:
                    alignments[i].remove(0)
                    alignments[i] = tuple(alignments[i])
                else:
                    alignments.pop(i)
                    i -= 1
                    break
    

            #else:
            i += 1


    return alignments
    


##### given an alignment, extract the relevant probabilities/parameters
## need 3 types: translation probs, fertility probs, distortion probs
### parameter structure:
#   translation:  (french, english) tuple, meaning the prob that english -> french
#   fertility: (english, fertility) tuple, meaning the prob that en | fertility val
#   distortion: (trg_pos, src_pos, trg_len)
trans_params = set()
fert_params = set()
distort_params = set()

def extract_params(french, english, alignment):
    to_return = [list(), list(), list()] # index 0 is for trans, 1 for fert, 2 for distort
    french = french.split()
    english = english.split()
    # note that length of alignment should be the same as English
    if len(english) != len(alignment):
        print(english)
        print(french)
        print(alignment)
        raise ValueError
    
    for i in range(len(alignment)):

        if alignment[i] == 0:
            # fertility = 0
            f = (english[i], 0)
            #to_return.append(f)
            fert_params.add(f)
            
        elif type(alignment[i]) == set:
            #fertility is > 1
            f = (english[i], len(alignment[i]))
            fert_params.add(f)
            to_return[1].append(f)
            for j in alignment[i]:
                t = (french[j-1], english[i])
                d = (j, i+1, len(french)) # distortion
                to_return[0].append(t)
                trans_params.add(t)
                to_return[2].append(d)
                distort_params.add(d)
                
            
        else: # pos is ordinary number
            f = (english[i], 1)
            t = (french[alignment[i]-1], english[i])
            d = (alignment[i], i+1, len(french))
            to_return[1].append(f)
            fert_params.add(f)
            to_return[0].append(t)
            trans_params.add(t)
            to_return[2].append(d)
            distort_params.add(d)

    return to_return



def init_probs(x): # x is the number of training examples to use
    # run through all sentences in data, then get normalized prob scores using length of sets
    with open('fr-en/processed-en.en','r') as f: #with open('fr-en/europarl-v7.fr-en.en','r') as f:
        en_lines = f.readlines()
    with open('fr-en/processed-fr.fr','r') as f: #with open('fr-en/europarl-v7.fr-en.fr','r') as f:
        fr_lines = f.readlines()
        
    c = 0
    for i in range(len(en_lines)):
        # keep all sentences under 6 words so as to avoid overly-expensive computation
        if len(en_lines[i].strip().split()) < 6 and len(fr_lines[i].strip().split()) < 6:
            print(en_lines[i].strip(), fr_lines[i].strip())
            possible_alignments = get_alignments(len(fr_lines[i].strip().split()), len(en_lines[i].strip().split()))
            print(len(possible_alignments))
            for align in possible_alignments:
                extract_params(fr_lines[i].strip(), en_lines[i].strip(), align)
            c += 1
            if c >= x: # x would be a parameter
                break

    # param sets should be completely filled
    trans = {}
    fert = {}
    distort = {}
    trans_prob = 1 / len(trans_params)
    fert_prob = 1 / len(fert_params)
    distort_prob = 1 / len(distort_params)
    for item in trans_params:
        trans[item] = trans_prob
    for item in fert_params:
        fert[item] = fert_prob
    for item in distort_params:
        distort[item] = distort_prob

    with open('params/init_trans_params.bn','wb') as f:
        #json.dump(trans, f)
        pickle.dump(trans, f)
    with open('params/init_fert_params.bn','wb') as f:
        #json.dump(fert, f)
        pickle.dump(fert, f)
    with open('params/init_distort_params.bn','wb') as f:
        #json.dump(distort, f)
        pickle.dump(distort, f)









'''
General Algorithm here:
Compute all possible alignments (and their probabilities) for each sentence in training.
Then once we have these, we calculate the expectation for each french/english pair by counting how many times each pair appears in an alignment and multiplying by the alignment probability each time;
for example, if a pair only appears once in each alignment it exists in, then the expectation would be the sum of the alignment probabilities that it exists in.
Once we have all of these expectations, we update the probability of each pair (f | e) by setting it equal to it's expectation over the sum of all other related pair expectations (f' | e).
Then we repeat for either a specified number of epochs or until convergence.
'''

def EM_Algo(training_num, epochs):

    '''
    # extract prior probabilities
    with open('params/init_distort_params_epoch_4.bn','rb') as f:#with open('params/init_distort_params.bn','rb') as f:
        distort_params = pickle.load(f)

    with open('params/init_fert_params_epoch_4.bn','rb') as f:#with open('params/init_fert_params.bn','rb') as f:
        fert_params = pickle.load(f)

    with open('params/init_trans_params_epoch_4.bn','rb') as f:#with open('params/init_trans_params.bn','rb') as f:
        trans_params = pickle.load(f)
    '''
    # initialize all probabilities to a single number to save time in initialization
    init_val = 0.0001
    distort_params = {}
    fert_params = {}
    trans_params = {}
    

    trans_expect = {}
    fert_expect = {}
    distort_expect = {}

    total_trans_expect = {}
    total_fert_expect = {}
    total_distort_expect = {}


    # load docs
    with open('fr-en/processed-en.en','r') as f:
        en_lines = f.readlines()
    with open('fr-en/processed-fr.fr','r') as f:
        fr_lines = f.readlines()


    def p_align(french, english, alignment): # calculates the probability of each alignment based on current params
        # multiply all the parameter probabilities together that contributed to the alignment
        params = extract_params(french, english, alignment)
        total_prob = 1
        t = params[0]
        f = params[1]
        d = params[2]
        for param in t:
            if param in trans_params:
                total_prob *= trans_params[param]
            else:
                total_prob *= init_val
        for param in f:
            if param in fert_params:
                total_prob *= fert_params[param]
            else:
                total_prob *= init_val
        for param in d:
            if param in distort_params:
                total_prob *= distort_params[param]
            else:
                total_prob *= init_val
        return total_prob

    for epoch in range(epochs):
        x = 0
        for i in range(len(en_lines)):
            if x >= training_num:
                break
            if len(en_lines[i].split()) < 6 and len(fr_lines[i].split()) < 6: # only small sentences to decrease computational overhead
                x += 1
                print(x,':',en_lines[i],',',fr_lines[i])
                alignments = get_alignments(len(fr_lines[i].strip().split()), len(en_lines[i].strip().split()))
                # calculate expectation of parameters
                # count how many times a given parameter appears across all alignments
                #trans_counts = {}
                #fert_counts = {}
                #distort_counts = {}
                #all_trans = list()
                #all_fert = list()
                #all_distort = list()
                for align in alignments:
                    params = extract_params(fr_lines[i], en_lines[i], align)
                    #all_trans.extend(params[0])
                    #all_fert.extend(params[1])
                    #all_distort.extend(params[2])

                    # make these params[] into sets, count how many times each parameter is in a given alignment
                    # then increase the global param expectation by #times parameter appears * alignment probability
                    align_prob = p_align(fr_lines[i], en_lines[i], align)
                    for j in set(params[0]):
                        if j not in trans_expect:
                            trans_expect[j] = params[0].count(j)*align_prob                        
                        else:
                            trans_expect[j] += params[0].count(j)*align_prob
                        if j[1] not in total_trans_expect:
                            total_trans_expect[j[1]] = params[0].count(j)*align_prob # want to extract only english from (french, english)
                        else:
                            total_trans_expect[j[1]] += params[0].count(j)*align_prob
                    for j in set(params[1]):
                        if j not in fert_expect:
                            fert_expect[j] = params[1].count(j)*align_prob
                        else:
                            fert_expect[j] += params[1].count(j)*align_prob
                        if j[0] not in total_fert_expect:
                            total_fert_expect[j[0]] = params[1].count(j)*align_prob # want to extract only english from (english, fertility)
                        else:
                            total_fert_expect[j[0]] += params[1].count(j)*align_prob
                    for j in set(params[2]):
                        if j not in distort_expect:
                            distort_expect[j] = params[2].count(j)*align_prob
                        else:
                            distort_expect[j] += params[2].count(j)*align_prob
                        if j[1:3] not in total_distort_expect:
                            total_distort_expect[j[1:3]] = params[2].count(j)*align_prob # want to extract only src_pos, trg_len from (trg_pos, src_pos, trg_len)
                        else:
                            total_distort_expect[j[1:3]] += params[2].count(j)*align_prob

        # now we have the expectation for all parameters in training!
        for param in trans_expect:
            if param in trans_expect and param[1] in total_trans_expect:
                trans_params[param] = trans_expect[param] / total_trans_expect[param[1]]
        for param in fert_expect:
            if param in fert_expect and param[0] in total_fert_expect:
                fert_params[param] = fert_expect[param] / total_fert_expect[param[0]]
        for param in distort_expect:
            if param in distort_expect and param[1:3] in total_distort_expect:
                distort_params[param] = distort_expect[param] / total_distort_expect[param[1:3]]

        # write the new parameters to a file
        with open('params/init_trans_params_epoch_'+str(epoch)+'.bn','wb') as f:
            pickle.dump(trans_params, f)
            #print(trans_params)
        with open('params/init_fert_params_epoch_'+str(epoch)+'.bn','wb') as f:
            pickle.dump(fert_params, f)
        with open('params/init_distort_params_epoch_'+str(epoch)+'.bn','wb') as f:
            pickle.dump(distort_params, f)
            


# for a given epoch of parameter values, find the top k most probable aligned word pairs
def most_probable_pairs(epoch, k):
    with open('params/init_trans_params_epoch_'+str(epoch)+'.bn','rb') as f:
        trans_params = pickle.load(f)
    #print(trans_params)
    to_return = list()
    for i in range(k):
        max_prob = 0
        max_param = None
        for param in trans_params:
            if trans_params[param] > max_prob:
                max_param = param
                max_prob = trans_params[param]
        #print(max_param, trans_params[max_param])
        to_return.append(max_param)
        del trans_params[max_param]
    return to_return


        
def translate(english, epoch):
    # NOTE: this translator is only able to translate into sequences <= the length of sequences found in training
    # for each word in english:
    #   find top fertility + top k most probable french words
    # now you have a french set of words for every English word
    # need to find the right alignment, so create every possible alignment and factor in distortion probabilities.
    # return most likely aligned sentence
    with open('params/init_trans_params_epoch_'+str(epoch)+'.bn','rb') as f:
        trans_params = pickle.load(f)
    with open('params/init_fert_params_epoch_'+str(epoch)+'.bn','rb') as f:
        fert_params = pickle.load(f)
    with open('params/init_distort_params_epoch_'+str(epoch)+'.bn','rb') as f:
        distort_params = pickle.load(f)

    indiv_trans = {}
    BOW = list()
    French_translation = list()
    french_len = 0
    for word in range(len(english.split())):
        # find max fertility probability
        max_prob = 0
        max_fert = 0
        for i in fert_params:
            if i[0] == english.split()[word] and fert_params[i] > max_prob: # word+1 since index starts at 1 for fertility probs
                if True:#i[1] < (len(english.split()) / (french_len+1)): # some formula ensures we don't overwhelm the model with sentences that are too long
                    max_fert = i[1]
                    max_prob = fert_params[i]
                    #print(i,fert_params[i])

        # now we have the fertility k, find the k most probable french words
        top_k = list()
        #print(english.split()[word],'has fertility',max_fert)
        recycle = {}
        for k in range(max_fert):
            max_prob = 0
            max_trans = None
            for pair in trans_params:
                if pair[1] == english.split()[word] and trans_params[pair] > max_prob:
                    max_prob = trans_params[pair]
                    max_trans = pair[0]
                    p = pair

            recycle[p] = max_prob
            del trans_params[p]
            #print(english.split()[word],max_trans)
            if max_fert==1:
                top_k = max_trans
                #print('...',max_trans, indiv_trans[word-1])
            elif max_trans not in top_k:
                top_k.append(max_trans)
                #print(':::',max_trans, indiv_trans[word-1])
        #top_k = #most_probable_pairs(epoch, max_fert)
        trans_params.update(recycle)
        BOW.extend(top_k)
        french_len += max_fert
        #print('MAX_FERT for word',english.split()[word],'is',max_fert)
        indiv_trans[word] = top_k

    #print(indiv_trans)
    #print(len(BOW),'for BOW')
    for word in range(len(english.split())):
        # if fert > 1: for each word in list place in top trg position
        # if word's fertility is 1: place in top distortion trg position
        if type(indiv_trans[word])==list and len(indiv_trans[word])>0:
            for i in indiv_trans[word]:
                # get the top distortion parameter
                max_prob = 0
                max_distort = None
                for j in distort_params:
                    if j[1] == word+1 and j[2] == len(BOW) and distort_params[j] > max_prob:
                        max_prob = distort_params[j]
                        max_distort = j
                if max_distort == None: # if we couldn't find a good parameter
                    max_distort = [word]
                else:
                    # now max_distort has the ideal position of this word
                    #print(max_distort)
                    del distort_params[max_distort] # delete so we don't get duplicate positions; since 'word' will change we don't need this param anymore
                #print('i:',i)
                French_translation.insert(max_distort[0],i)
        elif type(indiv_trans[word])!=list:
            # get the top distortion parameter
            max_prob = 0
            max_distort = None
            for i in distort_params:
                if i[1] == word and i[2] == len(BOW) and distort_params[i] > max_prob:
                    max_prob = distort_params[i]
                    max_distort = i[0]
            if max_distort == None:
                max_distort = word
            # now max_distort has the ideal position of this word
            French_translation.insert(max_distort, indiv_trans[word])
            
    # format output
    to_return = ''
    for term in French_translation:
        to_return += term+' '
    return to_return.strip()


def test(epoch):
    from nltk.translate.bleu_score import sentence_bleu
    with open('fr-en/test-set-en.en','r') as f:
        en_lines = f.readlines()
    with open('fr-en/test-set-fr.fr','r') as f:
        fr_lines = f.readlines()

    if len(en_lines) != len(fr_lines):
        print('test sets do not match!')
        raise ValueError

    scores = list()
    average_score = 0
    n_gram_weights = (1,0,0,0)
    for i in range(len(en_lines)):
        reference = fr_lines[i].strip()
        candidate = translate(en_lines[i].strip(), epoch)
        score = sentence_bleu(reference, candidate,weights=n_gram_weights)
        #print(score)
        scores.append(score)
        average_score += score
        # extra, optional test:
        #if not search_error(reference, candidate, en_lines[i], epoch):
        #    print('modeling error:\n\tEnglish:',en_lines[i],'\n\tReference:',reference,'\n\tTranslation:',candidate)

    # plotting stuff
    import matplotlib.pyplot as plt
    plt.plot(range(len(en_lines)), scores)
    plt.ylabel('BLEU scores')
    plt.show()



    return average_score / len(en_lines)
        


def search_error(french_gold, french_candidate, english, epoch):
    # extract prior probabilities
    with open('params/init_distort_params_epoch_'+str(epoch)+'.bn','rb') as f:#with open('params/init_distort_params.bn','rb') as f:
        distort_params = pickle.load(f)

    with open('params/init_fert_params_epoch_'+str(epoch)+'.bn','rb') as f:#with open('params/init_fert_params.bn','rb') as f:
        fert_params = pickle.load(f)

    with open('params/init_trans_params_epoch_'+str(epoch)+'.bn','rb') as f:#with open('params/init_trans_params.bn','rb') as f:
        trans_params = pickle.load(f)
    init_val = 0.0001
    def p_align(french, english, alignment): # calculates the probability of each alignment based on current params
        # multiply all the parameter probabilities together that contributed to the alignment
        params = extract_params(french, english, alignment)
        total_prob = 1
        t = params[0]
        f = params[1]
        d = params[2]
        for param in t:
            if param in trans_params:
                total_prob *= trans_params[param]
            else:
                total_prob *= init_val
        for param in f:
            if param in fert_params:
                total_prob *= fert_params[param]
            else:
                total_prob *= init_val
        for param in d:
            if param in distort_params:
                total_prob *= distort_params[param]
            else:
                total_prob *= init_val
        return total_prob

    alignments = get_alignments(len(french_candidate.split()), len(english.split()))
    most_prob_cand = 0
    for align in alignments:
        prob = p_align(french_candidate, english, align)
        if prob > most_prob_cand:
            most_prob_cand = prob

    alignments = get_alignments(len(french_gold.split()), len(english.split()))
    most_prob_gold = 0
    for align in alignments:
        prob = p_align(french_gold, english, align)
        if prob > most_prob_gold:
            most_prob_gold = prob

    if most_prob_gold >= most_prob_cand:
        print('search error')
        return True
    else:
        print('modeling or translation error')
        return False
    
def get_prob_pairs(word, k, epoch): # word is in English
    with open('params/init_trans_params_epoch_'+str(epoch)+'.bn','rb') as f:
        trans_params = pickle.load(f)
    with open('params/init_fert_params_epoch_'+str(epoch)+'.bn','rb') as f:
        fert_params = pickle.load(f)
    #with open('params/init_distort_params_epoch_'+str(epoch)+'.bn','rb') as f:
    #    distort_params = pickle.load(f)

    for i in range(k):
        max_prob = 0
        max_item = None
        for param in trans_params:
            if param[1]==word and trans_params[param] > max_prob:
                max_prob = trans_params[param]
                max_item = param
        if max_item != None:
            print(max_item, max_prob)
            del trans_params[max_item]

    for i in range(k):
        # getting fertilities
        max_prob = 0
        max_item = None
        for param in fert_params:
            if param[0]==word and fert_params[param] > max_prob:
                max_prob = fert_params[param]
                max_item = param
        if max_item != None:
            print(max_item, max_prob)
            del fert_params[max_item]



    

# FOR TESTING
#french = "J'aime les fleurs"
#english = "I like flowers"
#alignment = ({1,3}, 0, 2)
#probs = extract_params(french, english, alignment)
#print(probs)


#init_probs(20)
#print('INITIALIZED!')

#EM_Algo(1200, 5)
#print(most_probable_pairs(0, 20))
#print(most_probable_pairs(4, 20))
#print()
#print(most_probable_pairs(1, 15))
#print(translate('the debate is closed', 4))
#search_or_model_error("c'est important","pourquoi c'est c'est c'est important","this is an important matter",8)
print(test(4))
#get_prob_pairs('not',9,4)
#print()
#get_prob_pairs('the',9,4)
#print()
#get_prob_pairs('thank',4,4)
#print(translate('the numbers speak for themselves', 4))
#print(translate('exports would also suffer', 4))
#print(translate('please indicate your position', 4))
#print(translate('parliament adopted the resolution', 4))
