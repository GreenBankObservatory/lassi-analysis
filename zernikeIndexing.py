from copy import copy

"""
This is a simple module for converting between the different 
Zernike indexing standards
"""

# ansi indicies move left to right, down the pyramid
ansiZs = [   0,
            1, 2,
           3, 4, 5,
          6, 7, 8, 9,
        10, 11, 12, 13, 14,
      15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 26, 27,
  28, 29, 30, 31, 32, 33, 34, 35]

assert ansiZs == range(36)  

# here is how the noll indicies layout on the same pyramid
nollZs = [  1,   # n = 0 
           3, 2, # n = 1
          5, 4, 6,  # n = 2
        9, 7, 8, 10, # n = 3
      15, 13, 11, 12, 14, # n = 4
    21, 19, 17, 16, 18, 20, # n = 5
  27, 25, 23, 22, 24, 26, 27, # n = 6
35, 33, 31, 29, 30, 32, 34, 36] # n = 7

def printZs(zs):
    "Print out the list of like the pyramids above"

    lenZs = len(zs)

    row = 0
    rowLen = 1
    start = 0
    end = 1
    cnt = 0
    while end <= lenZs:
       print zs[start:end]
       row += 1
       rowLen += 1
       start = end
       end += rowLen

def noll2ansi(zs):
    "Converts NOLL ordering to ANSI ordering"

    # we need to support zs of generic type
    num = len(zs)
    ansis = copy(zs)

    for i, z in enumerate(zs):
        nollI = nollZs[i]
        # make sure we don't gag on inputs less then 36
        if nollI  < num:
            ansis[i] = copy(zs[nollI-1])

    return ansis


def ansi2noll(zs):
    "Converts ANSI ordering to NOLL ordering"

    # we need to support zs of generic type
    num = len(zs)
    noll = copy(zs)

    for i, z in enumerate(zs):
        ansiI = ansiZs[i]
        # make sure we don't gag on inputs less then 36
        if ansiI  < num:
            noll[i] = copy(zs[ansiI-1])

    return noll

def testNoll2ansi():
    #ansi = range(1, 37)
    #noll = ansi2noll(ansi)
    noll = range(1,37)
    ansi = noll2ansi(noll)
    print "ansi: "
    printZs(ansi)
    print "noll: "
    printZs(noll)
    assert ansi == nollZs

def main():
    testNoll2ansi()

if __name__=='__main__':
    main()
