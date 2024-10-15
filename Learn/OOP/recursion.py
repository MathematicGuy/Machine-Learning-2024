interest_rate = 0.02
money = 100

def money_interest(
        money: int,  # Initial amount of money
        interest_rate: float,  # Interest rate as a decimal (e.g., 0.05 for 5%)
        repeat_time: int,  # Number of times to apply the interest
        start_index: int  # Starting index for the loop
    ) -> int:

    loop = repeat_time
    start = start_index
    
    money = money + money*interest_rate
    print(f"loop {start}, total moneyoney: {money:.4f}")
    if start < loop:
        start += 1
        return money_interest(money, interest_rate, loop, start)

money_interest(money, interest_rate, 100, 0)
    