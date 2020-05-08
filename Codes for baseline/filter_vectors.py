import re
import json
def convertDict (vectorList, wordList, destination):
    dic_str = open(wordList).readline()
    dic = re.findall(r'<\w+>|\w+', dic_str)
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

        
        

convertDict("/Users/yinglu/Documents/grad_school/nlp/ChID-Dataset/data/glove.6B.200d.txt","/Users/yinglu/Documents/grad_school/nlp/ChID-Dataset/Codes for baseline/cloth_word_dict.txt", "/Users/yinglu/Documents/grad_school/nlp/ChID-Dataset/Codes for baseline/cloth_word_vectors.txt")
convertDict("/Users/yinglu/Documents/grad_school/nlp/ChID-Dataset/Codes for baseline/cloth_word_vectors.txt","/Users/yinglu/Documents/grad_school/nlp/ChID-Dataset/Codes for baseline/cloth_candidate_dict.txt", "/Users/yinglu/Documents/grad_school/nlp/ChID-Dataset/Codes for baseline/cloth_candidate_vectors.txt")
