from items import Item
from phone import Phone

item1 = Item("Phone", 100, 2)
item = Item("Laptoptoptop", 1000, 3)
item3 = Phone("Smartphone", 500, 5, 1) # 500*4=2000

# Item.instantiate_from_csv()
# print("total phone assest:", Phone.total_assets()) # only phone assets, not all items
print(Item.all)
# print("Total item price:", Item.total_item_price())

item.name = "Nikkeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee" # setting an attribute
print(item.name) # reading an attribute

# item.apply_increament(0.2) # capsulation
# print(item.price)
# #? capsulation: don't allow direct access to attribute directly but through a method
# item.apply_discount() 
# print(item.price)

item.send_email()

#? Polymorshism: single function can handle different kind of Object
