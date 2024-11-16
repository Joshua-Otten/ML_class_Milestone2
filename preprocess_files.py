with open('fr-en/europarl-v7.fr-en.en','r') as f:
        en_lines = f.readlines()
with open('fr-en/europarl-v7.fr-en.fr','r') as f:
        fr_lines = f.readlines()

# preprocessing step
count = 0
with open('fr-en/processed-en.en','w') as e:
        with open('fr-en/processed-fr.fr','w') as f:
                for i in range(len(en_lines)):
                        if len(en_lines[i].split()) < 6 and len(fr_lines[i].split()) < 6:
                                count += 1
                                #print(en_lines[i])
                                en_lines[i] = ' '.join(' '.join(' '.join(' '.join(' '.join(' '.join(' '.join((' '.join(en_lines[i].strip().lower().split('.'))).split(',')).split('!')).split('(')).split(')')).split('-')).split('?')).split(':'))
                                #print(en_lines[i])
                                #print('%%%%%%%%%%%%%%%%%%%%')
                                fr_lines[i] = ' '.join(' '.join(' '.join(' '.join(' '.join(' '.join(' '.join((' '.join(fr_lines[i].strip().lower().split('.'))).split(',')).split('!')).split('(')).split(')')).split('-')).split('?')).split(':'))
                                #print(fr_lines[i])
                                if en_lines[i].strip() != '' and fr_lines[i].strip() != '':
                                        e.write(en_lines[i].strip()+'\n')
                                        f.write(fr_lines[i].strip()+'\n')
print(count)


# create test set
with open('fr-en/processed-en.en','r') as f:
        en_lines = f.readlines()
with open('fr-en/processed-fr.fr','r') as f:
        fr_lines = f.readlines()

with open('fr-en/test-set-en.en','w') as e:
        with open('fr-en/test-set-fr.fr','w') as f:
                for i in range(len(en_lines)-73, len(en_lines)):
                        e.write(en_lines[i].strip()+'\n')
                        f.write(fr_lines[i].strip()+'\n')
# then just take out the examples from the training set files
