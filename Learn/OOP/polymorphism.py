from abc import abstractmethod
class Animal:
    @abstractmethod
    #? Each animal type can override the sound() method to provide its own implementation
    def sound():
        pass        
    
def animal_sound(animal):
    return f'{animal.__class__.__name__} make sound: {animal.sound()}'    

class Dog(Animal):
    def sound(self):
        return "Bark Bark"    

class Cat(Animal):
    def sound(self):
        return "Meow Meow"

zoo = [Dog(), Cat()]

for animal in zoo:
    print(animal_sound(animal))
