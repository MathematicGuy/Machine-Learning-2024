class Animal:
    all = []
    
    def __init__(self, sound):
        self.sound = sound
        Animal.all.append(self)
    
    def make_sound(self):
        return f'{self.__class__.__name__} make sound: {self.sound}'
    
    
dog = Animal("Bark Bark")
cat = Animal("Meow Moew")
for ani in Animal.all:
    print(ani.make_sound())
    