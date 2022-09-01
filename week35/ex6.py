import random as rand

class Fraction:
    def __init__(self, x, y):
        self.numerator = x
        self.denominator = y
        print(x)
        print(y)

    def print(self):
        print(f"The fractional number is: {self.numerator}/{self.denominator}")
        return

    
    def simplify(self):
        gcd = GCD(self.numerator, self.denominator)
        print(f"Reduced number is: {int(self.numerator/gcd)}/{int(self.denominator/gcd)}")
        return


def GCD(x, y):
    if y == 0: return x
    return GCD(y, x%y)


if __name__ == "__main__":
    while True:
        choice = input("Give choice:\n\
            1: Random numbers\n\
            2: Choose numbers\n\
            9: Quit: ")

        match choice:
            case '1':
                while True:
                    try:
                        min_num=int(input("Give min value of range: "))
                        max_num=int(input("Give max value of range: "))
                    except ValueError:
                        print("Please use integers!")
                        continue
                    else:
                        number=Fraction(
                            rand.randrange(min_num, max_num), rand.randrange(min_num, max_num)
                            )
                        number.print()
                        number.simplify()
                        break
            case '2':
                while True:
                    try:
                        num=int(input("Give numerator: "))
                        denom=int(input("Give denominator: "))
                    except ValueError:
                        print("Please use integers!")
                        continue
                    else:
                        number=Fraction(num, denom)
                        number.print()
                        number.simplify()
                        break
            case '9':
                print("Thank you have a nice day!")
                break
            case _: 
                print("Invalid choice try again!")
                