
from items import Item

#? Inheritance
class Phone(Item): # inherit Item class
    def __init__(self, name: str, price: float, quantity=0, broken_phone=0):
        #? Instance Level
        #? Call to super func to have access to all attributes/methods from mother class: Item   
        super().__init__(
            name, price, quantity
        )

        # Run validations to the received arguments
        assert broken_phone >= 0, f"Broken Phone {broken_phone} is not greater than or equal to zero"
        self.broken_phone = broken_phone
                
    # class method allow total_assets to be called on the class itself, thus not calculating all instances from mother class         
    @classmethod
    def total_assets(cls):
        total = 0
        for instance in cls.all:
            if isinstance(instance, cls):
                total += instance.price * (instance.quantity - instance.broken_phone)
                
        return total

if __name__ == '__main__':
    phone1 = Phone("Ha", 244.4, 4, 1) # instance/object
    print(phone1.name)
    
    