import gcld3

detector = gcld3.NNetLanguageIdentifier(min_num_bytes=0, 
                                        max_num_bytes=1000)


def predict(filename):
    file_r  = open(filename, 'r')
    lines = file_r.readlines()
    file_r.close()

    file_w = open('langues.txt', 'w')

    for line in lines :
        result = detector.FindLanguage(text=line)
        file_w.write(result.language + ', ')
        file_w.write(str(result.probability) + '\n')
        
    
    file_w.close()

predict('homotopie_tatoeba2.txt')
