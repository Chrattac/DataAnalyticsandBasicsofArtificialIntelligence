obey = False;
while(not obey):
    try:
        a=int(input("Give number: "))
        b=int(input("Give another number: "))
    except ValueError:
        print("Please enter only numbers")
        continue
    else:
        obey = True;

        if a > b:
            print(f'Number {a} is bigger than {b}')
        elif b > a:
            print(f'Number {b} is bigger than {a}')
        else:
            print('Numbers are equal')

