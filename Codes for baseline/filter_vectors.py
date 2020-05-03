def convertDict (vectorList, wordList, destination):
    dic_str = open(wordList).readline()
    dic = eval(dic_str)
    keyFile = open(vectorList, 'r')
    writeFile = open(destination, 'w')
    lines = keyFile.readlines()
    for line in lines:
        line.strip("\n")
        vec = line.split()
        word = vec[0]
        if word in dic:
            writeFile.writelines(line)
    keyFile.close()
    writeFile.close()

        
        

convertDict("../data/glove.6B.200d.txt","cloth_word_dict.txt", "cloth_word_vectors.txt")
convertDict("cloth_word_vectors.txt","cloth_candidate_dict.txt", "cloth_candidate_vectors.txt")
