# Learn String Concatenation and String Formatting
# Few examples of string concatenation and string formatting
# print("space " + "X")
# print("space{}".format("X"))
# print(f"space{'X'}")


adj = input("Adjective: ")
noun = input("Noun: ")
verb = input("Verb: ")

madlibs = f"{noun} is so {adj}! It makes me so excited all the time because I love to {verb}."
print(madlibs)