import random
import unicodedata
import string
import numpy as np
import sentencepiece as spm

all_letters = string.ascii_letters + " .,;'-?!_"

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

def processString(txt):
  dictionary = {"'": " '", ";": " ;", ".": " .", ",": " ,"}
  transTable = txt.maketrans(dictionary)
  txt = txt.translate(transTable)
  return txt

def fusion(filename1, filename2):
    lines1 = readLines(filename1)

    lines2 = readLines(filename2)

    file = open('fusion_file.txt', 'w')
    for line in lines1:
        line = processString(line)

        file.write(line+'\n')

    for line in lines2:
        line = processString(line)
        file.write(line+'\n')

    file.close()


def minus(filename):
    file  = open(filename, 'r')
    lines = file.readlines()
    file.close()
    new_file = open('minus_file.txt', 'w')
    for Line in lines:
        line = Line.lower()
        new_file.write(line)
    new_file.close()

#minus('test.txt')

def process(filename):
    lines = readLines(filename)
    new_file = open('process_file.txt', 'w')
    for line in lines:
        line = processString(line)
        line = line.lower()
        new_file.write(line+'\n')
        
    new_file.close()

def text_to_spm(filename):
    file = open(filename, 'r')
    lines = file.readlines()
    file.close()
    sp = spm.SentencePieceProcessor(model_file = 'sp_model.model')

    file = open(filename,'w')
    for line in lines:
        sp_line = sp.encode(line, out_type =str)
        for word in sp_line[0:-1]:
            file.write(word)
        file.write('\n')
    file.close()
    

def split_test_train_valid(filename, n_train = 10000, n_test=10000, length = 50, n_valid = 10000 ):
    file  = open(filename, 'r')
    lines = file.readlines()
    file.close()
    random.shuffle(lines)

    train_file = open('tatoeba.train.txt', 'w')
    idx=0
    while idx<n_train:
        
        train_file.write(lines[idx])
        idx += 1
    train_file.close()

    valid_file = open('tatoeba.valid.txt', 'w')
    while idx<n_train + n_valid:
        if len(lines[idx])>=length:
            valid_file.write(lines[idx])
        idx += 1
    valid_file.close()

    test_file = open('tatoeba.test.txt', 'w')
    while idx<n_train + n_test + n_valid:
        if len(lines[idx])>=length:
            test_file.write(lines[idx])
        idx+=1
    test_file.close()

    

def split_test_train_valid2(filename1, filename2, n_train = 10000, n_test=10000, length = 50, n_valid = 10000 ):
    file  = open(filename1, 'r')
    lines1 = file.readlines()
    file.close()

    file  = open(filename2, 'r')
    lines2 = file.readlines()
    file.close()
    
    lst_idx = [i for i in range(len(lines1))]
    
    random.shuffle(lst_idx)
    
    train_file = open('tatoeba.train.txt', 'w')
    train_file1 = open('tatoeba.train.fr.txt', 'w')
    idx=0
    while idx<n_train:
        
        train_file1.write(lines1[lst_idx[idx]])
        train_file.write(lines1[lst_idx[idx]])
        idx += 1
    train_file1.close()

    train_file2 = open('tatoeba.train.en.txt', 'w')
    idx=0
    while idx<n_train:
        
        train_file2.write(lines2[lst_idx[idx]])
        train_file.write(lines2[lst_idx[idx]])
        idx += 1
    train_file2.close()
    train_file.close()

    idx = n_valid
    valid_file1 = open('tatoeba.valid.fr.txt', 'w')
    while idx<n_train + n_valid:
        if len(lines1[lst_idx[idx]])>=length:
            valid_file1.write(lines1[lst_idx[idx]])
        idx += 1
    valid_file1.close()

    idx = n_valid
    valid_file2 = open('tatoeba.valid.en.txt', 'w')
    while idx<n_train + n_valid:
        if len(lines2[lst_idx[idx]])>=length:
            valid_file2.write(lines2[lst_idx[idx]])
        idx += 1
    valid_file2.close()

    idx = n_test
    test_file1 = open('tatoeba.test.fr.txt', 'w')
    while idx<n_train + n_valid + n_test:
        if len(lines1[lst_idx[idx]])>=length:
            test_file1.write(lines1[lst_idx[idx]])
        idx += 1
    test_file1.close()

    idx = n_test
    test_file2 = open('tatoeba.test.en.txt', 'w')
    while idx<n_train + n_valid +n_test:
        if len(lines2[lst_idx[idx]])>=length:
            test_file2.write(lines2[lst_idx[idx]])
        idx += 1
    test_file2.close()

    text_to_spm('tatoeba.train.fr.txt')
    text_to_spm('tatoeba.train.en.txt')
    text_to_spm('tatoeba.test.fr.txt')
    text_to_spm('tatoeba.test.en.txt')
    text_to_spm('tatoeba.valid.fr.txt')
    text_to_spm('tatoeba.valid.en.txt')
    text_to_spm('tatoeba.train.txt')



#fusion('Tatoeba.en-fr.fr.txt', 'Tatoeba.en-fr.en.txt')
#minus('fusion_file.txt')
#split_test_train_valid('minus_file.txt')

#process('Tatoeba.en-fr.en.txt')

split_test_train_valid2('Tatoeba.fr.txt', 'Tatoeba.en.txt', n_train = 10000, n_test=10000, length = 50, n_valid = 10000 )
