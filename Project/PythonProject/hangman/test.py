# # import pytest
# from hangman import get_valid_word, word_to_blank, replace_blank, hangman

# def test_get_valid_word():
#     words = ["naruto", "one piece", "attack-on-titan", "bleach"]
#     word = get_valid_word(words)
#     assert '-' not in word and ' ' not in word

# def test_word_to_blank():
#     word = "naruto"
#     blank_word = word_to_blank(word)
#     assert blank_word == ["_", "_", "_", "_", "_", "_"]

# def test_replace_blank():
#     guess = 'a'
#     player_word_list = ["_", "_", "_", "_", "_", "_"]
#     word_list = list("naruto")
#     new_word_list = replace_blank(guess, player_word_list, word_list)
#     assert new_word_list == ["_", "a", "_", "_", "_", "_"]

# def test_hangman(monkeypatch):
#     word = "naruto"
#     inputs = iter(['n', 'a', 'r', 'u', 't', 'o'])
#     monkeypatch.setattr('builtins.input', lambda _: next(inputs))
#     assert hangman(word) == False

# if __name__ == "__main__":
#     # pytest.main() # type: ignore
import torch

bboxes = torch.tensor([
    [0, 0.95, 100, 100, 200, 200],  # Class 0, high confidence
    [0, 0.90, 110, 110, 210, 210],  # Class 0, slightly lower confidence, overlaps with previous box
    [1, 0.85, 50, 50, 150, 150],    # Class 1, different region
    [0, 0.80, 120, 120, 220, 220],  # Class 0, overlaps with first two boxes
    [1, 0.75, 55, 55, 155, 155],    # Class 1, overlaps slightly with another box in the same class
    [2, 0.60, 300, 300, 400, 400],  # Class 2, no overlap with others
    [2, 0.50, 310, 310, 410, 410],  # Class 2, overlaps with another box in the same class
])

for box in bboxes:
    print(box[2:])