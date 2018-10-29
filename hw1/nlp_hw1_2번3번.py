
# coding: utf-8

# In[1]:

global queue
global depth
global crude_dict
global output_string
crude_dict = []
output_string = ''


# In[2]:

class Node(object):
    
    def __init__(self, char):
        
        self.children = []
        self.pos = []
        self.syllable = char        
        self.is_leaf = 0 
        


# In[3]:

def split_line(line):
    
    worPLUSpos = line.replace(' ', '+').replace('\n', '+').split('+')
    for i in range( len(worPLUSpos) ):        
        temp = worPLUSpos[i].split('/')
        crude_dict.append([temp[0], temp[1]])


# In[4]:

def find(node, key):
    
    temp_node = node
    if not node.children:
        return None
    
    for char in key:
        can_not_find = True
        for child in temp_node.children:
            if child.syllable == char:
                can_not_find = False
                temp_node = child
                break
        if can_not_find:
            return None
    
    return temp_node.pos


# In[5]:

def insert(root, string, pos):

    node = root
    counter = 0
    
    for char in string:
        found_in_child = False
        for child in node.children:
            if child.syllable == char:
                node = child
                found_in_child = True
                break
        if not found_in_child:
            new_node = Node(char)
            node.children.append(new_node)
            node = new_node
            counter += 1
            
    node.is_leaf = 1
    
    if pos not in node.pos:
        node.pos.append(pos)
        
    return counter


# In[6]:

def make_trie(root):
    for i in range(len(crude_dict)):  
        #insert(root, "나", "NP")
        insert(root, crude_dict[i][0], crude_dict[i][1])    
        
    


# In[7]:

with open("morph_rule.txt") as f:
    for i in f.readlines():
        split_line(i.splitlines()[0])


# In[8]:

root = Node('*')
make_trie(root)


# In[9]:


queue = []
depth = []
depth.append(0)

def output(root):
    
    f = open("morph_dict.txt", "a")
    node = root
    
    for child in node.children:
        if child.is_leaf:
            for i in range(depth[0]):
                pop_value = queue.pop()
                print(pop_value, end='', file=f)
                depth[0] -= 1
            if queue:
                depth[0] += 1
            print(child.syllable, ' ', child.pos, file=f)
            if child.children:
                for j in range(len(child.children)):
                    queue.insert(0, child.syllable)
                depth[0] += 1
                output(child)
            
        else:
            for i in range(len(child.children)):
                queue.insert(0, child.syllable)
            depth[0] += 1
            output(child)    


# In[10]:

output(root)


# In[11]:

global gram_check
gram_check = []

def split_line_for_grammar(line): 
    
    worPLUSpos = line.split()
    for i in range( len(worPLUSpos) ):  
        if '+' in worPLUSpos[i]:
            gram_rule = worPLUSpos[i].replace('/', '+').split('+')
            for j in range(len(gram_rule)-1, -1, -1):
                if j % 2 == 0:
                    del(gram_rule[j])
            for j in range(len(gram_rule)-1, 0, -1):
                gram_rule.insert(j, '+')
            gram_rule_str = ''.join(gram_rule)
            gram_check.append(gram_rule_str)

def check(morps):
    for i in range(len(gram_check)):
        if morps in gram_check[i]:
            return True
    
    return False

with open("morph_rule.txt") as f:
    for i in f.readlines():
        split_line_for_grammar(i.splitlines()[0])


# In[12]:

def split_line_for_TP(line):
    
    worPLUSpos = line.split()
    for i in range( len(worPLUSpos) ):  
        TP(worPLUSpos[i])


# In[13]:

def TP(word):
    
    T = [] #T 매트릭스 생성
    T_mor = []
    T_out = [] #출력용
    for i in range(len(word)):
        temp = []
        temp_mor = []
        temp_out = []
        for j in range(len(word)):
            temp.append([])
            temp_mor.append([])
            temp_out.append([])
        T.append(temp)
        T_mor.append(temp_mor)
        T_out.append(temp_out)
    
    point = 0
    for i in range(len(word)):
        for j in range(len(word)-i):                        
            T[i][j].append(word[point+j])
        point += 1
    
    for i in range(len(word)):
        for j in range(1, len(word)-i):
            combine = []
            combine.append(T[i][j-1][0])            
            combine.append(T[i][j][0])
            combine_str = ''.join(combine)
            T[i][j].pop()
            T[i][j].append(combine_str)
    
    #사전 및 형태소 문법 검사
    for i in range(len(word)):
        for j in range(i+1):                        
            str_pos = find(root, T[len(word)-i-1][j][0])################################################################pos
            if str_pos:
                for k in range(len(str_pos)):
                    T_mor[len(word)-i-1][j].append(str_pos[k])
                    T_out[len(word)-i-1][j].append(T[len(word)-i-1][j][0]+ '/'+ str_pos[k])#@@@@
            if i != j:
                for e in range(i):
                    for l in range(len(T_mor[len(word)-i-1][j])):
                        for m in range(len(T_mor[len(word)-e-1][e])):
                            tester = (T_mor[len(word)-i-1][j][l] + '+' + T_mor[len(word)-e-1][e][m])
                            if check(tester) == True and len(T[len(word)-i-1][i][0]) == len(T[len(word)-i-1][j][0]) + len(T[len(word)-e-1][e][0]):
                                T_mor[len(word)-i-1][i].append(tester)
                                T_out[len(word)-i-1][i].append(T_out[len(word)-i-1][j][l] + '+' + T_out[len(word)-e-1][e][m])
            
    for i in range(len(T_out[0][len(word)-1])):
        if i == len(T_out[0][len(word)-1]) -1:
            print(T_out[0][len(word)-1][i], end='     ')
        else:
            print(T_out[0][len(word)-1][i], 'or ', end='')


# In[14]:

with open("input.txt") as f:
    for i in f.readlines():
        split_line_for_TP(i.splitlines()[0]) #line하나를 보냄
        print()


# In[15]:

"""
<참고문헌>
http://nlp.sogang.ac.kr/zb/download.php?id=2018_nlp_course&page=1&sn1=&divpage=1&sn=off&ss=on&sc=on&select_arrange=headnum&desc=asc&no=3&filenum=1
http://cs.kangwon.ac.kr/~leeck/NLP/05_morp.pdf
https://en.wikipedia.org/wiki/Trie
http://blog.ilkyu.kr/entry
"""

