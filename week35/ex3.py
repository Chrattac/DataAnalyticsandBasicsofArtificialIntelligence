import random as rand

a=rand.randrange(100)
b=rand.randrange(100)

if a > b:
    print(f'Number a {a} is bigger than b {b}')
elif b > a:
    print(f'Number b {b} is bigger than a {a}')
else:
    print(f'Numbers are equal, number a was {a} and b {b}')

