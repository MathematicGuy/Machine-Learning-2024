import random
from words import anime_words

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
    for index, i in enumerate(word_list): # use enumerate to get current index
        if guess == i: # check if each letter in word_list equal to guess 
            correct_indexs.append(index) # save correct guess index in correct_indexes
    
    # replace "_" with correct guesses
    for i in correct_indexs:
        new_word_list[i] = guess         
    
    return new_word_list


def hangman(word):
    '''
        Hangman is a game where you guess a letter of a word until you guessed all the letter or running out of lives 
        + word: word for player to guess
        + player_word_list ['_','_','_'] are a list of '_' (blank) created base on word length. each letter in word is a '_' in player_word_list
        + lives: live of the player
        
        The game main loop logic are:
            + keep track of guessed_word (word that been guess)
            + keep track of player progress by replacing blanks with correct guess (a letter) in player_word_list 
            + deduced live for incorrect guess
            + check losing/winning condition
            
        input: word (word to guess)
        output: true/false (continue playing or not)
    '''
    lives = 6
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"
    guessed_word = []

    word_list = list(word) # turn word to list: naruto = ['n', 'a', 'r', 'u', 't', 'o']
    player_word_list = word_to_blank(word)
    alive = True
    
    while alive:
        guess = input("Your Guess: ").lower()
        
        # Check if word are already guessed
        if guess in guessed_word:
            print("You already guessed this letter")
            lives -= 1    
        # Check if the guess is right
        else:    
            guessed_word.append(guess)        
            if guess in word_list:
                print("Correct")
                
                # Replace bank with correct guess in player_word_list by index
                player_word_list = replace_blank(guess, player_word_list, word_list)                
            else:
                if len(guess) > 1 or guess not in alphabet:
                    print("Invalid Guess, Please enter a 'single' letter from the alphabet")
                    lives -= 1    
                    continue # skip the loop
                else:
                    print("You guess wrong")
                lives -= 1
                
            
        #? Every Turn
        # Print out current live and guess every turn            
        print(f"You got {lives} lives left")
        print(f"Guesses: {player_word_list}")
        

        # Check Winning condition every guess
        if "_" not in player_word_list:
            play = input("You Won, wanna play more? y/n: ")
            if play == 'y':
                return True
            else:
                print("Thank for Playing !!!")
                return False
        # Check Losing condition every guess
        elif lives == 0:    
            print(f"You lost! The word was '{''.join(word_list)}'")
            play = input("Want to restart? y/n")
                                    
            if play == 'y':
                return True
            else:
                print("Thank for Playing !!!")
                return False
        

        # Seperate each turn with a Space for Authentic     
        print("")
        
        
play = True
while play:
    word = get_valid_word(anime_words).lower() # case sensitity
    print('psst the word is:', word)
    
    play = hangman(word)
