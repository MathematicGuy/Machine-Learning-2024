import csv

#? Initiate Class
class Item:
    #? Class Level
    pay_rate = 0.8 # discount 20%
    all = []
    
    def __init__(self, name: str, price: float, quantity=0):
        #? Instance Level
        # Run validations to the received arguments
        assert price >= 0, f"Price {price} is not greater than or equal to zero" 
        assert quantity >= 0, f"Quantity {quantity} is not greater than or equal to zero"
        
        # Assign to self object. Encapsulation
        self.__name = name # ''__' annotate private attribute, prevent access to the attribute from outside of the class
        self.__price = price
        self.quantity = quantity
        
        # Action to execute
        Item.all.append(self)

    @classmethod
    def instantiate_from_csv(cls):
        with open('items.csv', 'r') as f:
            items = csv.DictReader(f)
            
            for item in items:
                print(Item (
                    name = item.get('name'),
                    price = float(item.get('price')),
                    quantity = int(item.get('quantity'))
                ))
    
    @property
    # Property Decorator = Read-Only-Attribute
    def name(self):
        return self.__name
    
    @name.setter
    def name(self, value): # parameter set as value
        if len(value) > 25:
            raise Exception("Only accept name length under 25 letters")
        else:
            self.__name = value
    
    @property
    def price(self):
        return self.__price
    
    # def total_price(self):
    #     return self.__price * self.quantity
    
    def apply_discount(self):
        self.__price = self.__price * self.pay_rate
    
    def apply_increament(self, increase_value):
        self.__price = self.__price + self.__price*increase_value
    
    @classmethod
    def total_item_price(cls):
        total = 0
        for item in cls.all:
            total +=  item.price * item.quantity
        return total 
    
    @staticmethod
    def is_integer(num):
        if isinstance(num, float):
            return num.is_integer() # return True if the float a integer
        elif isinstance(num, int):
            return True
        else:
            return False

    # modify class output format (how this class be print out)
    def __repr__(self):
        # __class__.__name__ : access class name (through special function)
        return f"{self.__class__.__name__}({self.name}, {self.price}, {self.quantity})"


    def __connect(self, smtp_server):
        pass
    
    def __prepare_body(self):
        return f"""
            Hello
            We have {self.name} {self.quantity} times.
            Regards, Thanh
        """

    def __send(self):
        pass

    def send_email(self):
        self.__connect('')
        self.__prepare_body()
        self.__send()


if __name__ == '__main__':
    item1 = Item("Phone", 100, 2)
    item2 = Item("Phone2", 100, 2)

    Item.pay_rate = 0.5
    item1.apply_discount()
    item2.apply_discount()

    print(item1.price)
    print(item2.price)