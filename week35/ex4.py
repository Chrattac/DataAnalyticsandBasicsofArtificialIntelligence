import random as rand

def sum(x, y):
    return f'The sum of numbers {x} and {y} is {x+y}'

num1 = rand.randrange(10)
num2 = rand.randrange(10)

print(sum(num1, num2))