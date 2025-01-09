from models import User, Account, Transaction, ThanhBank, ThienBank, DefaultBank
from helpers import DataHandler

class BankSystem:
    def __init__(self):
        self.users = {}  # Key: email, Value: User object
        self.banks = {}  # Key: bank_name, Value: Bank object
        self.accounts = {}  # Key: (email, bank_name), Value: Account object
        self.load_data()
        self.current_user = None
        self.current_account = None

    def load_data(self):
        # Load users
        users_data = DataHandler.read_users()
        for user_dict in users_data:
            user = User(**user_dict)
            self.users[user.email] = user

        # Load banks
        banks_data = DataHandler.read_csv('banks.csv', ['bank_name', 'interest_rate', 'asset'])
        for bank_dict in banks_data:
            bank_name = bank_dict['bank_name']
            interest_rate = float(bank_dict['interest_rate'])
            asset = float(bank_dict['asset'])
            if bank_name == "SouthBank":
                bank = ThanhBank(bank_name, interest_rate)
            elif bank_name == "NorthBank":
                bank = ThienBank(bank_name, interest_rate)
            else:
                bank = DefaultBank(bank_name, interest_rate)  # Use DefaultBank
            bank.asset = asset
            self.banks[bank_name] = bank


        # Load accounts
        accounts_data = DataHandler.read_csv('assets.csv', ['email', 'bank_name', 'balance'])
        for acc_dict in accounts_data:
            email = acc_dict['email']
            bank_name = acc_dict['bank_name']
            balance = float(acc_dict['balance'])
            user = self.users.get(email)
            bank = self.banks.get(bank_name)
            if user and bank:
                account = Account(user, bank, balance)
                self.accounts[(email, bank_name)] = account

    def save_data(self):
        # Save users
        users_data = [{'email': user.email, 'name': user.name, 'password': user.password} for user in self.users.values()]
        DataHandler.write_users(users_data)

        # Save banks
        banks_data = [{'bank_name': bank.bank_name, 'interest_rate': bank.interest_rate, 'asset': bank.asset} for bank in self.banks.values()]
        DataHandler.write_csv('banks.csv', ['bank_name', 'interest_rate', 'asset'], banks_data)

        # Save accounts
        accounts_data = [{'email': account.user.email, 'bank_name': account.bank.bank_name, 'balance': account.balance} for account in self.accounts.values()]
        DataHandler.write_csv('assets.csv', ['email', 'bank_name', 'balance'], accounts_data)

    def run(self):
        self.show_welcome_message()
        self.login_or_signup()
        while True:
            self.select_bank()
            self.perform_operations()
            if not self.prompt_continue():
                break
        self.save_data()

    def show_welcome_message(self):
        print("Welcome to Our Bank System")

    def login_or_signup(self):
        while True:
            choice = input("\n1) Log In\n2) Sign Up\nChoose an option: ")
            if choice == '1':
                if self.login():
                    break
            elif choice == '2':
                self.signup()
            else:
                print("Invalid option. Please choose 1 or 2.")

    def login(self):
        email = input("Enter your email: ")
        password = input("Enter your password: ")
        user = self.users.get(email)
        if user and user.password == password:
            self.current_user = user
            print(f"\nWelcome back, {user.name}!")
            return True
        else:
            print("Invalid email or password. Please try again.")
            return False

    def signup(self):
        email = input("Enter your email: ")
        if email in self.users:
            print("Email already exists. Please log in.")
            return
        name = input("Enter your name: ")
        password = input("Enter your password: ")
        new_user = User(email, name, password)
        self.users[email] = new_user
        print("Sign up successful!")

    def select_bank(self):
        print("\nAvailable Banks:")
        bank_names = list(self.banks.keys())
        for idx, bank_name in enumerate(bank_names, start=1):
            print(f"{idx}) {bank_name}")
        choice = input("Choose a bank to continue: ")
        if choice.isdigit() and 1 <= int(choice) <= len(bank_names):
            bank_name = bank_names[int(choice) - 1]
            bank = self.banks[bank_name]
            self.current_account = self.get_or_create_account(self.current_user, bank)
            print(f"You are now banking with {bank_name}.")
        else:
            print("Invalid choice. Try again.")
            self.select_bank()

    def get_or_create_account(self, user, bank):
        account_key = (user.email, bank.bank_name)
        account = self.accounts.get(account_key)
        if not account:
            account = Account(user, bank)
            self.accounts[account_key] = account
        return account

    def perform_operations(self):
        while True:
            print("\n1) Deposit")
            print("2) Withdraw")
            print("3) Transaction")
            print("4) Display Bank Slogan")
            print("5) Money Interest Monthly")
            print("6) Change Name")
            print("7) Logout")
            choice = input("Choose an option: ")
            if choice == '1':
                self.handle_deposit()
            elif choice == '2':
                self.handle_withdraw()
            elif choice == '3':
                self.handle_transaction()
            elif choice == '4':
                self.display_bank_slogan()
            elif choice == '5':
                self.calculate_interest()
            elif choice == '6':
                self.change_name()
            elif choice == '7':
                print("Logging out of bank...")
                break
            else:
                print("Invalid option. Please choose a valid number.")

    def handle_deposit(self):
        amount = float(input("Enter amount to deposit: "))
        self.current_account.deposit(amount)
        print(f"Deposited ${amount}. New balance: ${self.current_account.balance}")

    def handle_withdraw(self):
        amount = float(input("Enter amount to withdraw: "))
        self.current_account.withdraw(amount)
        print(f"New balance: ${self.current_account.balance}")

    def handle_transaction(self):
        # Step 1: List all banks
        print("\nAvailable Banks:")
        bank_names = list(self.banks.keys())
        for idx, bank_name in enumerate(bank_names, start=1):
            print(f"{idx}) {bank_name}")
        bank_choice = input("Choose the receiver's bank by entering the number: ")
        if bank_choice.isdigit() and 1 <= int(bank_choice) <= len(bank_names):
            receiver_bank_name = bank_names[int(bank_choice) - 1]
            receiver_bank = self.banks[receiver_bank_name]
        else:
            print("Invalid choice. Transaction cancelled.")
            return

        # Step 2: List all users in the chosen bank
        receivers_in_bank = []
        for (email, bank_name), account in self.accounts.items():
            if bank_name == receiver_bank_name and email != self.current_user.email:
                user = self.users.get(email)
                if user:
                    receivers_in_bank.append((email, user.name))

        if not receivers_in_bank:
            print("No users found in that bank.")
            return

        # Step 3: List out all users and allow user to select one
        print("\nUsers in the selected bank:")
        for idx, (email, name) in enumerate(receivers_in_bank, start=1):
            print(f"{idx}) {name} ({email})")

        receiver_choice = input("Choose the receiver by entering the number: ")
        if receiver_choice.isdigit() and 1 <= int(receiver_choice) <= len(receivers_in_bank):
            receiver_email, receiver_name = receivers_in_bank[int(receiver_choice) - 1]
            receiver_user = self.users.get(receiver_email)
            receiver_account = self.get_or_create_account(receiver_user, receiver_bank)
        else:
            print("Invalid choice. Transaction cancelled.")
            return

        # Step 4: Proceed with the transaction
        try:
            amount = float(input("Enter amount to send: "))
            if amount <= 0:
                print("Amount must be positive.")
                return
        except ValueError:
            print("Invalid amount entered.")
            return

        transaction = Transaction(self.current_account, receiver_account, amount)
        if transaction.process():
            # Log transaction
            DataHandler.write_csv(
                'transactions.csv',
                ['sender_email', 'receiver_email', 'amount', 'sender_bank', 'receiver_bank'],
                [{
                    'sender_email': self.current_user.email,
                    'receiver_email': receiver_email,
                    'amount': amount,
                    'sender_bank': self.current_account.bank.bank_name,
                    'receiver_bank': receiver_bank_name
                }],
                append=True  # Now supported
            )
            print(f"Transaction successful! Sent ${amount} to {receiver_name} at {receiver_bank_name}.")
        else:
            print("Transaction failed due to insufficient funds.")



    def display_bank_slogan(self):
        slogan = self.current_account.bank.bank_slogan()
        print(f"Bank Slogan: {slogan}")

    def calculate_interest(self):
        months = int(input("Enter number of months: "))
        rate = self.current_account.bank.interest_rate
        for month in range(1, months + 1):
            interest = self.current_account.balance * rate
            self.current_account.balance += interest
            self.current_account.bank.asset += interest
            print(f"Month {month}: New balance: ${self.current_account.balance:.2f}")

    def change_name(self):
        new_name = input("Enter your new name: ")
        self.current_user.name = new_name
        print("Name updated successfully.")

    def prompt_continue(self):
        choice = input("Do you want to continue banking? (yes/no): ")
        return choice.lower() == 'yes'

