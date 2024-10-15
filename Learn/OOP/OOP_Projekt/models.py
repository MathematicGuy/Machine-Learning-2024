from abc import ABC, abstractmethod

class User:
    def __init__(self, email, name, password):
        self.email = email
        self.name = name
        self.password = password

class Bank(ABC):
    def __init__(self, bank_name, interest_rate):
        self.bank_name = bank_name
        self.interest_rate = interest_rate
        self.asset = 0

    @abstractmethod
    def bank_slogan(self):
        pass

    def deposit(self, amount):
        self.asset += amount

    def withdraw(self, amount):
        self.asset -= amount

class ThanhBank(Bank):
    def bank_slogan(self):
        return "Trust me bro"

class ThienBank(Bank):
    def bank_slogan(self):
        return "Don't Trust Bro"

class DefaultBank(Bank):
    def bank_slogan(self):
        return "Welcome to Default Bank!"

class Account:
    def __init__(self, user, bank, balance=0):
        self.user = user
        self.bank = bank
        self.balance = balance

    def deposit(self, amount):
        self.balance += amount
        self.bank.deposit(amount)

    def withdraw(self, amount):
        if amount <= self.balance:
            self.balance -= amount
            self.bank.withdraw(amount)
        else:
            print("Insufficient funds.")

class Transaction:
    def __init__(self, sender_account, receiver_account, amount):
        self.sender_account = sender_account
        self.receiver_account = receiver_account
        self.amount = amount

    def process(self):
        if self.sender_account.balance >= self.amount:
            self.sender_account.withdraw(self.amount)
            self.receiver_account.deposit(self.amount)
            print(f"Transferred ${self.amount} from {self.sender_account.user.name} to {self.receiver_account.user.name}")
            return True
        else:
            print("Insufficient funds for transaction.")
            return False
