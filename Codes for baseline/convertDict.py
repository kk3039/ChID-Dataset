def convertDict (source, destination):
    keyFile = open(source, 'r')
    key = eval(keyFile.readline())[:100000]
    responseFile = open(destination, 'w')
    for id,token in enumerate(key):
        responseFile.writelines(token+"\n")

convertDict("./wordList.txt","dict.txt")
