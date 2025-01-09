from abc import abstractmethod

class Bank:
    bank_asset = 0
        
    def __init__(self, bank_name, interest_rate):
        self.bank_name = bank_name
        self.__interest_rate = interest_rate 
        
        
    @property 
    def interest_rate(self):
        return self.__interest_rate

    @abstractmethod
    def bank_slogan():
        pass
    
    #! Static Method
    @staticmethod
    def conversion_rate(currency1, currency2, amount):
        if currency1 == "vnd" and currency2 == "usd":
            return amount*0.000043  # Example conversion rate
        else:
            return "we don't have that currency yet"
    
    
    @interest_rate.setter
    def interest_rate(self, value):
        #? modify class attribute 'interest_rate'
        self.__interest_rate = value
    
    def __repr__(self):
        return f"{self.__class__.__name__} bank with {self.__bank_asset} and {self.interest_rate}"

def bank_slogan(bank):
    return f'{bank.__class__.__name__} slogan is "{bank.bank_slogan()}"'


class ThanhBank(Bank): #! Inheritant
    def __init__(self, bank_name, interest_rate):
        super().__init__(bank_name, interest_rate)
            
    #! Polymorphism
    def bank_slogan(self):
        return "Trust me bro"
    

class ThienBank(Bank): #! Inheritant
    def __init__(self, bank_name, interest_rate):
        super().__init__(bank_name, interest_rate)
            
    #! Polymorphism
    def bank_slogan(self):
        return "Don't Trust Bro"
    

class Account:
    total_account_asset = 0
    accounts = []
    
    def __init__(self, name, email, account_asset):
        self.__name = name
        self.__email = email
        self.account_asset = account_asset
        Account.accounts.append(self)
        
    #! Getter and Setter
    #? Getter/Setter for name
    #! Encapsulation
    @property 
    def name(self):
        return self.__name
    
    @name.setter
    def name(self, value):
        if len(value) > 30:
            raise ValueError("Your Name is too long, must be less than 30 letters")
        else:
            self.__name = value
    
    #? Getter/Setter for email 
    #! Encapsulation
    @property
    def email(self):
        return self.__email
    
    @email.setter
    def email(self, value):
        self.__email = value
    
    def withdraw(self, amount):
        if amount <= self.account_asset:
            self.account_asset -= amount
        else:
            print(f"Current Asset: ${self.account_asset}")
            print(f"You trying to withdraw ${amount}, You don't have that much money")
    
    #? Do this last
    # def transaction(user, cash):
    #     account_asset -= cash
        
    def deposit(self, cash):
        self.account_asset += cash
        print(f"{self.__name} have deposit ${self.account_asset}")
        
    
    def money_interest_monthly(
        self,
        interest_rate: float,  # Interest rate as a decimal (e.g., 0.05 for 5%)
        months: int,  # Number of times to apply the interest
        start_index: int,  # Starting index for the loop
        bank_name: str
    ) -> int:

        loop = months
        start = start_index
        
        self.account_asset = self.account_asset + self.account_asset*interest_rate
        print(f"Month {start}th, asset: ${self.account_asset:.4f}")
        if start < loop:
            start += 1
            return self.money_interest_monthly(interest_rate, loop, start, bank_name) # type: ignore
    
    
    #! Class Method: allow access to class attribute
    @classmethod
    def cal_total_account_asset(cls):
        total = 0
        for account in cls.accounts:
            total += account.account_asset
            
        print(f'total account assets: {total} with {interest_rate} interest rate at {bank_name}')
    
    
    def __repr__(self):
        return f"{self.__class__.__name__} \n name:{self.name}, \n asset:{self.interest_rate}"
    

print("Welcome to Our Bank System")
#? User
User1 = Account("Thanh", "dinhnhatthanh248@gmail.com", 0)
User1.deposit(100)
User2 = Account("Thien", "thien@gmail.com", 0)
User2.deposit(300)

#? Bank
# User1.withdraw(300)
Bank1 = Bank("SouthBank", 0.02)
Bank2 = Bank("NortBank", 0.02)

amount = 2_000_000
print(f'{amount} vnd to dollar convert to: {Bank.conversion_rate("vnd", "usd", amount)}')

#? Test Function
interest_rate = Bank1.interest_rate
input_months = 100
bank_name = Bank1.bank_name
#? Expected Assets base on monthly interest
# User1.money_interest_monthly(interest_rate, input_months, 0, bank_name)

#? All User Asset Combine
Account.cal_total_account_asset()

#? Test Polymorphism
banks = [ThanhBank("VeryThanh", 0.4), ThienBank("VeryThien", 0.123)]
for bank in banks:
    print(bank_slogan(bank))
