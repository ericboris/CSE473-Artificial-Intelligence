from typing import List, Tuple, Dict, Callable
from collections import defaultdict
import re


def is_multiple_of_9(n: int) -> bool:
    '''Return True if n is a multiple of 9; False otherwise.'''
    return n % 9 == 0


def next_prime(m: int) -> int:
    '''Return the first prime number p that is greater than m.
    You might wish to define a helper function for this.
    You may assume m is a positive integer.'''
    x = m + 1
    while not isPrime(x):
        x += 1
    
    return x


def isPrime(n: int) -> bool:
    ''' Return True if n is prime and False otherwise.'''
    if n <= 3:
        return n > 1
    elif n % 2 == 0 or n % 3 == 0:
        return False
    else:
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True

def next_prime(m: int) -> int:
    return 0

def sieve(n: int) -> int:

def naturalNums
        

def quadratic_roots(a: int, b: int, c: int) -> Tuple[float, float] or str:
    '''Return the roots of a quadratic equation (real cases only).
    Return results in tuple-of-floats form, e.g., (-7.0, 3.0)
    Return 'complex' if real roots do not exist.'''
    det = (b ** 2) - (4 * a * c)
    if det < 0:
        return 'complex'
    
    plus = (-b + det ** 0.5) / (2 * a)
    minus = (-b - det ** 0.5) / (2 * a)
    
    return (plus, minus)


def perfect_shuffle(even_list: List[int]) -> List[int]:
    '''Assume even_list is a list of an even number of elements.
    Return a new list that is the perfect-shuffle of the input.
    For example, [0, 1, 2, 3, 4, 5, 6, 7] => [0, 4, 1, 5, 2, 6, 3, 7]'''
    result = []
    n = len(even_list) // 2
    
    for i in range(n):
        result.append(even_list[i])
        result.append(even_list[i + n])
    
    return result


def triples_list(input_list: List[int]) -> List[int]:
    '''Assume a list of numbers is input. Using a list comprehension,
    return a new list in which each input element has been multiplied
    by 3.'''
    return [x * 3 for x in input_list]


def double_consonants(text: str) -> str:
    '''Return a new version of text, with all the consonants doubled.
    For example:  "The *BIG BAD* wolf!" => "TThhe *BBIGG BBADD* wwollff!"
    For this exercise assume the consonants are all letters OTHER
    THAN A,E,I,O, and U (and a,e,i,o, and u).
    Maintain the case of the characters.'''
    VOWELS = ['A', 'E', 'I', 'O', 'U']
    
    result = []
    
    for ch in text:
        result.append(ch)
        if ch.isalpha() and ch.upper() not in VOWELS:
            result.append(ch)
    
    return ''.join(result)


def count_words(text: str) -> dict:
    '''Return a dictionary having the words in the text as keys,
    and the numbers of occurrences of the words as values.
    Assume a word is a substring of letters and digits and the characters
    '-', '+', '*', '/', '@', '#', '%', and "'" separated by whitespace,
    newlines, and/or punctuation (characters like . , ; ! ? & ( ) [ ]  ).
    Convert all the letters to lower-case before the counting.'''
    
    CHARS = {'-', '+', '*', '/', '@', '#', '%', "'"}
    isValid = lambda ch : ch.isalnum() or ch in CHARS
    
    text = text.lower()
    
    # Let splitWords be the words in text
    # with the punctuation removed.
    splitWords = []
    
    # Loop over text adding words to 
    # splitWords and splitting on 
    # punctuation.
    w = ''
    for ch in text:
        if isValid(ch):
            w += ch
        else:
            if len(w):
                splitWords.append(w)
                w = ''
    
    # Make sure that the last word
    # is added to splitWords.
    if len(w):
        splitWords.append(w)
    
    # Let wordCounts be the dictionary
    # that holds the counts of each word
    # in text.    
    wordCounts = {}
    
    # Count the words in text.
    for w in splitWords:
        if w not in wordCounts:
            wordCounts[w] = 0
        wordCounts[w] += 1
    
    return wordCounts
    

    

def make_cubic_evaluator(a: float, b: float, c: float, d: float) -> Callable[[float], float]:
    '''When called with 4 numbers, returns a function of one variable (x)
    that evaluates the cubic polynomial
    a x^3 + b x^2 + c x + d.
    For this exercise your function definition for make_cubic_evaluator
    should contain a lambda expression.'''
    return lambda x: (a * x ** 3) + (b * x ** 2) + (c * x) + d


class Polygon:
    """Polygon class."""
    def __init__(self, n_sides: int, lengths: List[int] = None, angles: List[int] = None):
        self.n_sides = n_sides
        self.lengths = lengths
        self.angles = angles
    
    def is_rectangle(self) -> bool:
        ''' Return True if the polygon is a rectangle,
        False if it is definitely not a rectangle, and None
        if the angle list is unknown (None).'''
        if self.n_sides != 4:
            return False
        if self.lengths != None and (len(self.lengths) != 4 or self.lengths[0] != self.lengths[2] or self.lengths[1] != self.lengths[3]):
            return False
        if not self.angles:
            return None
        if len(self.angles) != 4 or any([a != 90 for a in self.angles]):
            return False
        return True
    
    
    def is_rhombus(self) -> bool:
        if self.n_sides != 4:
            return False
        if self.lengths != None and (len(self.lengths) != 4 or any([s != self.lengths[0] for s in self.lengths])):
            return False
        if self.angles != None and (len(self.angles) != 4 or self.angles[0] != self.angles[2] or self.angles[1] != self.angles[3]):
            return False
        if not self.lengths:
            return None
        return True
    
    
    def is_square(self) -> bool:
        if self.n_sides != 4:
            return False
        if self.angles != None and (len(self.angles) != 4 or any([a != 90 for a in self.angles])):
            return False
        if self.lengths != None and (len(self.lengths) != 4 or any([k != self.lengths[0] for k in self.lengths])):
            return False
        if self.lengths == None:
            return None
        if self.angles == None:
            return
        return True
    
    
    def is_regular_hexagon(self) -> bool:
        if self.n_sides != 6:
            return False
        if self.lengths != None and (len(self.lengths) != 6 or any([k != self.lengths[0] for k in self.lengths])):
            return False
        if self.angles != None and (len(self.angles) != 6 or any([a != 120 for a in self.angles])):
            return False
        if not self.angles or not self.lengths:
            return False
        return True
    
    
    def is_isosceles_triangle(self) -> bool:
        if self.n_sides != 3:
            return False
        if self.lengths == None and self.angles == None:
            return None
        if self.lengths != None and (len(self.lengths) != 3 or len(set(self.lengths)) > 2):
            return False
        if self.angles != None and (len(self.angles) != 3 or len(set(self.angles)) > 2):
            return False
        return True
    
    
    def is_equilateral_triangle(self) -> bool:
        if self.n_sides != 3:
            return False
        if self.lengths == None and self.angles == None:
            return None
        if self.lengths != None and (len(self.lengths) != 3 or len(set(self.lengths)) != 1):
            return False
        if self.angles != None and (len(self.angles) != 3 or len(set(self.angles)) != 1):
            return False
        return True
    
    
    def is_scalene_triangle(self) -> bool:
        if self.n_sides != 3:
            return False
        if self.lengths == None and self.angles == None:
            return None
        if self.lengths != None and (len(self.lengths) != 3 or len(set(self.lengths)) != 3):
            return False
        if self.angles != None and (len(self.angles) != 3 or len(set(self.angles)) != 3):
            return False
        return True

    
    
    
    
    
    
    
    
    