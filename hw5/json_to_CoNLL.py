
# coding: utf-8

# In[1]:


import sys,json,re


# In[2]:

def getMorphs(wordO, morphO):#lemma와 pos를 반환해주는 목적이다. 또한 ne를 구하기 위해 mid와 wid값을 같이 반환한다.
    #이 때, mid는 morph의 id를 의미하고 wid는 word의 id+1를 의미한다.
    mBegin = int(word['begin'])
    mEnd = int(word['end'])
    morphs = []
    for morphIndex in range(mBegin, mEnd+1):
        morph = morphO[morphIndex]
        morphs.append({'lemma':morph['lemma'], 'mid':morph['id'], 'pos':morph['type'], 'wid':wordO['id']+1})
    return morphs


# In[3]:

def appendNEs(morphs, neO):
    #B(Begin)은 target phrase의 새로운 시작 단어
    #I(Inside)는 target phrase의 일부분이나 처음이 아닌 단어
    #O(Other)는 target phrase가 아닌 단어
    for aMorph in morphs:
        for aNe in neO:
            #morph의 id가 ne의 id(index) 사이에 해당하는 경우 B또는 I를 의미한다.
            if aNe['begin'] <= aMorph['mid'] and aNe['end'] >= aMorph['mid']:
                if aNe['begin'] == aMorph['mid']:
                    #morph의 id가 ne의 시작 id(index)와 같다면 B이고
                    aMorph['ne'] = 'B_%s'%aNe['type']#B인 경우 ne를 그대로 붙여준다.                
                else:
                    #moprh의 id가 ne의 시작 id(index)가 아니라면 I이다.
                    aMorph['ne'] = 'I'
    for aMorph in morphs:#B, I에 해당되지 않는 나머지 morph들은 모두 O가 된다.
        if not 'ne' in aMorph:
            aMorph['ne'] = 'O'
    return morphs


# In[4]:

if __name__=='__main__':
    
    names = ['train', 'dev', 'NEtaggedCorpus_test (1)']#2016KLPexpo_NERcorpus.zip에 존재하는 3개의 json 파일을 target으로 한다.
    
    for name in names:
        jO = json.loads(open(name+'.json', 'r', encoding = 'UTF8').read())#json 파일을 읽어온다.
        writer = open(name+'.txt', "w")
        sens = jO['sentence']
        for sen in sens:
            for word in sen['word']:
                morphs = getMorphs(word, sen['morp'])#morps에 lemma와 pos를 불러온다.
                morphs = appendNEs(morphs, sen['NE'])#morps에 ne를 불러온다.
                for morph in morphs:
                    lemma = morph['lemma']
                    ne = morph['ne']
                    wid = morph['wid']
                    pos = morph['pos']
                    writer.write( lemma+'/'+pos+' '+ne+'\n' )#형태소 원형'lemma', 품사'pos', 개체명'ne'를 txt파일에 출력해준다.
            writer.write('\n')
        writer.close()


# In[ ]:



