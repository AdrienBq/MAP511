import random
import unicodedata
import string

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


fusion('Tatoeba.en-fr.fr.txt', 'Tatoeba.en-fr.en.txt')
minus('fusion_file.txt')
split_test_train_valid('minus_file.txt')