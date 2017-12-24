import numpy.random as npr
npr.seed(123)
my_length = 100
bp = ['A','C','G','T']
DNA = ''.join(npr.choice(bp,size=my_length))


def count_val(dna,val):
    count = 0
    for i in dna:
        if i == val:
           count += 1
    return count
count_val(DNA,'A')  

def count_val_2(dna,seq):
    count = 0
    for i in range(len(dna)):
        if ''.join(dna[i:(i+2)]) == seq:
            count += 1
    return count
count_val_2(DNA,'AA')  

def count_val_3(dna,seq):
    count = 0
    n = len(seq)
    for i in range(len(dna)):  
        if ''.join(dna[i:(i+len(seq))]) == seq: 
            count += 1
            pat = ''.join(dna[i:(i+len(seq))])
            print 'pat:'+ pat    
    return count

count_val_3(DNA,'AAAA') 



npr.seed(123)
dna_length = 10000
bp = ['A','C','G','T']
newDNA = ''.join(npr.choice(bp,size=dna_length))

def count_val_4(dna,seq): 
    
    n = 0
    
    pos_list = [n for n in xrange(len(dna)) if dna.find(seq,n) == n]

    return pos_list
    
count_val_4(newDNA,'ACAGA') 

npr.seed(123)
dna_length = 100000
bp = ['A','C','G','T']
DNA_1 = ''.join(npr.choice(bp,size=dna_length))
DNA_2 = ''.join(npr.choice(bp,size=dna_length))

def count_val_5(dna1,dna2,seq): 
    
     n = 0 
     list1 =  [n for n in xrange(len(dna1)) if dna1.find(seq, n) == n] 
     list2 = [n for n in xrange(len(dna2)) if dna2.find(seq, n) == n]
     return len(set(list2).intersection(list1))
                   
count_val_5(DNA_1,DNA_2,'ACA')