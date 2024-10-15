# import pytest
from hangman import get_valid_word, word_to_blank, replace_blank, hangman

def test_get_valid_word():
    words = ["naruto", "one piece", "attack-on-titan", "bleach"]
    word = get_valid_word(words)
    assert '-' not in word and ' ' not in word

def test_word_to_blank():
    word = "naruto"
    blank_word = word_to_blank(word)
    assert blank_word == ["_", "_", "_", "_", "_", "_"]

def test_replace_blank():
    guess = 'a'
    player_word_list = ["_", "_", "_", "_", "_", "_"]
    word_list = list("naruto")
    new_word_list = replace_blank(guess, player_word_list, word_list)
    assert new_word_list == ["_", "a", "_", "_", "_", "_"]

def test_hangman(monkeypatch):
    word = "naruto"
    inputs = iter(['n', 'a', 'r', 'u', 't', 'o'])
    monkeypatch.setattr('builtins.input', lambda _: next(inputs))
    assert hangman(word) == False

if __name__ == "__main__":
    pytest.main()