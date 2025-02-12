import random

def generate_dice_rolls(n, seed=0):
    random.seed(seed) # set a fixed randomness each run
    return [random.randint(1, 6) for _ in range(n)]

n_rolls = 1000
dice_rolls = generate_dice_rolls(n_rolls)

def count_occurrences(dice_rolls, number):
    count = 0
    
    for roll in dice_rolls:
        if roll == number:
           count += 1
    
    return count 


def cal_prob(dice_rolls, number):
    total = len(dice_rolls)
    count = count_occurrences(dice_rolls, number)
    
    return count / total
    


number_of_interest = 4
occurrences = count_occurrences(dice_rolls, number_of_interest)
print(occurrences)
prob = cal_prob(dice_rolls, number_of_interest)
print(prob)