import random as rand

num_arr_1 = []
num_arr_2 = []
ans_arr = []
usr_arr = []
correct = 0

for i in range(0,5):
    obey = False
    num_arr_1.append(rand.randrange(10))
    num_arr_2.append(rand.randrange(10))
    ans_arr.append(num_arr_1[i]*num_arr_2[i])
    while (not obey):
        try:
            ans = int(input(f'{num_arr_1[i]} × {num_arr_2[i]} = '))
        except ValueError:
            print("Please give number as an anwer!")
            continue
        else:
            obey=True
            usr_arr.append(ans)

for i in range(0,5):
    if(usr_arr[i] == ans_arr[i]):
        print('Correct! :-) – {x} × {y} = {z}'
        .format(
            x=num_arr_1[i], y=num_arr_2[i], z=ans_arr[i]))
        correct+=1
    else:
        print('Incorrect :-( – {x} × {y} = {z}'
        .format(
            x=num_arr_1[i], y=num_arr_2[i], z=ans_arr[i]))

print(f"You got {correct} correct!")