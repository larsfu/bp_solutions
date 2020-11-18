#!/usr/bin/env python3
import numpy as np

######
# a) #
######
text = "Hello DELTA!"
print(text)
# The `with` statement automatically closes the file after writing is done.
with open("build/file1.txt", "w") as f:
    f.write(text)


######
# b) #
######
# Note that np.arange(a, b) and range(a, b) generate numbers in the half-open
# [a, b) – we need to set 101 to get 100 as the last entry.
array = np.arange(1, 101)
print(array)
# np.savetxt saves a numpy array to a text file in a human-readable format.
# The fmt argument specifies the number formatting, as in general
# there are many ways of representing a number in text.
np.savetxt("build/file2.txt", array, fmt="%i")


######
# c) #
######
# This so-called “decorator” allows this function to be applied to a numpy array.
@np.vectorize
# The function isprime therefore only needs to check the prime-ness of one number.
def isprime(a):
    # A number is a prime when it is larger than one and all division remainders up
    # to the number itself are non-zero
    return a > 1 and all(a % i for i in range(2, a))
    #                    ^^^^^^^^^^^^^^^^^^^^^^^^^^
    # This is called a “list comprehension” and is an short and easy way to generate
    # a list in python. It contains all remainders up to the number being checked.
    #
    # The `all` function returns False if any of the numbers in the are zero, else True.


# np.genfromtxt allows to read data from a human-readable file like the one we saved before.
numbers = np.genfromtxt("build/file2.txt", dtype=int)
print(numbers[isprime(numbers)])
#             ^^^^^^^^^^^^^^^^
# The return value if this is called a mask. It contains a True/False value for every
# item in the list it refers to. When you execute `array[mask]`, you get back only
# the list items that had True as their mask entry.
