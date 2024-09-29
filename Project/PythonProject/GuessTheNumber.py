import random

max = random.randint(70, 200)
num = random.randint(1, max)
print("pssst the number is: ", num)
NotGuessed = True
while NotGuessed:
    try:
        guess = int(input(f"Guess the number between 1 and {max}: "))
        if guess == num:
            print("You guessed it!")
            NotGuessed = False
        elif abs(guess - num) > 0 and abs(guess - num) < 10:
            print("You getting closer")
        elif guess > num:
            print("lower")
        elif guess < num:
            print("higher")
    except ValueError:
        print("pls guess a number")
    