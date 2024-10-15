
from items import Item


class keyboard(Item): # inherit Item class
    def __init__(self, name: str, price: float, quantity=0):
        #? Instance Level
        # Call to super func to have access to all attributes/methods from mother class: Item   
        super().__init__(
            name, price, quantity
        )

      

if __name__ == '__main__':
    keyboard1 = keyboard("Ha", 244.4, 4)
    print(keyboard1.name)