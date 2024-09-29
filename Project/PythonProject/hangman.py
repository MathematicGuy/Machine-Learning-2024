import random
from words import anime_words

aphabet = "abcdefghijklmnopqrstuvwxyz"
guessed_word = []

def get_valid_word(words):
    word = random.choice(words)  # Randomly chooses a word from the list
    # 
    while '-' in word or ' ' in word:
        word = random.choice(words)
    
    return word.lower()
        
def word_to_blank(word):
    list_word = []
    for i in word:
        list_word.append("_")

    return list_word

def replace_blank(guess, player_word_list, word_list):
    # Replace bank with correct guess in player_word_list by index
    #? If 1 guess have 2 correct letter, then replace 2 correct blank with that letter
    new_word_list = player_word_list # copy player_word_list 
    correct_indexs = [] # save correct guess index
    for i in word_list:
        print("ai:", i)
        if guess == i: # check if each letter in word_list equal to guess 
            correct_indexs.append(word_list.index(i)) # save correct guess index in correct_indexes
    
    # replace "_" with correct guesses
    for i in correct_indexs:
        new_word_list[i] = guess 
    print("Correct index: ", correct_indexs)

    
    return new_word_list

def hangman(word):
    lives = 6
    word_list = list(word) # turn word to list: naruto = ['n', 'a', 'r', 'u', 't', 'o']
    player_word_list = word_to_blank(word)
    alive = True
    
    while alive:
        guess = input("Your Guess: ")

        # Check if word are already guessed
        if guess in guessed_word:
            print("You already guessed this letter")
        # Check if the guess is right
        else:
            if "_" not in player_word_list:
                alive = False
            
            guessed_word.append(guess)        
            if guess in word_list:
                print("Correct")
                
                # Replace bank with correct guess in player_word_list by index
                player_word_list = replace_blank(guess, player_word_list, word_list)
                print(f"Guesses: {player_word_list}")
            else:
                lives -= 1
                print("You guess wrong")
                print(f"You got {lives} lives left")
                
        print("")
        
        
play = True
while play:
    word = get_valid_word(anime_words)
    print('psst the word is:', word)
    live = True
    
    hangman(word)
        