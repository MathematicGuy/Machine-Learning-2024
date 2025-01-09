import csv
import os

class DataHandler:
    data_folder = 'data/'

    @staticmethod
    def read_csv(filename, fieldnames):
        data = []
        filepath = os.path.join(DataHandler.data_folder, filename)
        if os.path.exists(filepath):
            with open(filepath, mode='r', newline='') as file:
                reader = csv.DictReader(file, fieldnames=fieldnames)
                next(reader)  # Skip header
                for row in reader:
                    data.append(row)
        return data

    @staticmethod
    def write_csv(filename, fieldnames, data, append=False):
        filepath = os.path.join(DataHandler.data_folder, filename)
        mode = 'a' if append else 'w'
        file_exists = os.path.exists(filepath)
        with open(filepath, mode=mode, newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            if not file_exists or mode == 'w':
                writer.writeheader()
            writer.writerows(data)

    @staticmethod
    def read_users():
        fieldnames = ['email', 'name', 'password']
        return DataHandler.read_csv('users.csv', fieldnames)

    @staticmethod
    def write_users(users):
        fieldnames = ['email', 'name', 'password']
        DataHandler.write_csv('users.csv', fieldnames, users)

    @staticmethod
    def read_banks():
        fieldnames = ['bank_name', 'interest_rate', 'asset']
        return DataHandler.read_csv('banks.csv', fieldnames)

    @staticmethod
    def write_banks(banks):
        fieldnames = ['bank_name', 'interest_rate', 'asset']
        DataHandler.write_csv('banks.csv', fieldnames, banks)

    @staticmethod
    def read_accounts():
        fieldnames = ['email', 'bank_name', 'balance']
        return DataHandler.read_csv('assets.csv', fieldnames)

    @staticmethod
    def write_accounts(accounts):
        fieldnames = ['email', 'bank_name', 'balance']
        DataHandler.write_csv('assets.csv', fieldnames, accounts)

    @staticmethod
    def read_transactions():
        fieldnames = ['sender_email', 'receiver_email', 'amount', 'sender_bank', 'receiver_bank']
        return DataHandler.read_csv('transactions.csv', fieldnames)

    @staticmethod
    def write_transactions(transactions, append=False):
        fieldnames = ['sender_email', 'receiver_email', 'amount', 'sender_bank', 'receiver_bank']
        DataHandler.write_csv('transactions.csv', fieldnames, transactions, append=append)
