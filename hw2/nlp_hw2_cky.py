
# coding: utf-8

# In[1]:

global gram_rule
gram_rule = {}


# In[2]:

def split_line_gram(gram_rule, line):
    
    line = line.replace('\n', '.')
    line = line.replace('->','.')
    line = line.replace(' ', '.')
    rules = line.split('.')
    for i in range(len(rules)):
        if '' in rules:
            rules.remove('')
    if rules[0] not in gram_rule:
        gram_rule[rules[0]] = []
    add = []
    for i in range(len(rules)-1):
        add.append(rules[i+1])
    if add not in gram_rule[rules[0]]:
        gram_rule[rules[0]].append(add)


# In[3]:

with open("grammar.txt", 'r', encoding = 'utf-8-sig') as f:
    for line in f:
        if line == '\n':
            continue
        split_line_gram(gram_rule, line)


# In[4]:

def CKY(words, grammar):
    
    f = open("output.txt", 'a')
    
    table = []
    table_out = []
    for i in range(len(words)):
        component = []
        component_out = []
        for j in range(len(words)+1):
            component.append([])
            component_out.append([])
        table.append(component)
        table_out.append(component_out)
    for i in range(1, len(words)+1):
        for nter, ter in gram_rule.items():
            if [words[i-1]] in ter:
                table[i-1][i].append(nter)
                print(nter,'->',words[i-1])###################printing works
                print(nter,'->',words[i-1], file = f)###################printing works
    
    for i in range(1, len(words)+1):
        for nter, ter in gram_rule.items():
            for j in range(len(table[i-1][i])):
                if [table[i-1][i][j]] in ter:
                    add_string = '(' + nter + ' ' + words[i-1] + ')'
                    print(nter,'->',table[i-1][i][j])#################printing works
                    print(nter,'->',table[i-1][i][j], file = f)#################printing works
                    table_out[i-1][i].append(add_string)
                    table[i-1][i].remove(table[i-1][i][j])
                    table[i-1][i].append(nter)
    
    for j in range(2, len(words)+1):
        for i in range(j-2, -1, -1):
            for k in range(i+1, j):
                for m in range(len(table[i][k])):
                    for n in range(len(table[k][j])):
                        for nter, ter in gram_rule.items():
                            for h in range(len(ter)):
                                if [table[i][k][m],table[k][j][n]] == ter[h]: #or [table[k][j][n],table[i][k][m]] == ter[h]:
                                    table[i][j].append(nter)
                                    table_out[i][j].append('(' + nter +' '+table_out[i][k][m]+table_out[k][j][n]+')')
                                    print(nter,'->',table[i][k][m],'',table[k][j][n])######################printing works
                                    print(nter,'->',table[i][k][m],'',table[k][j][n], file = f)######################printing works
    
    
    
    for i in range(len(table[0][len(words)])):
        if table[0][len(words)][i] == 'S':
            print(table_out[0][len(words)][i])##############print works
            print(table_out[0][len(words)][i], file =f)##############print works
    
    print("", file = f)##############print works    
    f.close()
        
        


# In[5]:

def split_line(line):    
    words = line.split()
    CKY(words, gram_rule)#<-----------------CKY의 시작


# In[10]:

with open("input.txt", 'r', encoding = 'utf-8-sig') as f:
    for line in f:
        if '.' in line:
            line = line.replace('.', '')
        if '\n' in line:
            line = line.replace('\n', '')
        print(line)
        print(line[len(line)-1])
        split_line(line)
        print("")
        


# In[ ]:




# In[ ]:



