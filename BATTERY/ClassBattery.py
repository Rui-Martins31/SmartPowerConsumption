class Battery:
    """
    Simulated battery
    """
    def __init__(self, 
                 battery_total_capacity: float, 
                 battery_curr_capacity: float = 0.0,
                 charger_rate: float = 7.2):
        self.curr_capacity: float = battery_curr_capacity
        self.total_capacity: float = battery_total_capacity
        if charger_rate <= self.total_capacity/8:
            self.charger_rate: float = charger_rate
        else:
            self.charger_rate: float = self.total_capacity/8


        if self.total_capacity <= 0.0:
            raise ValueError
        if self.curr_capacity < 0.0:
            raise ValueError
        
    def charge(self, amount_kwh: float = None) -> float:
        """
        Charges battery and returns the amount of kWh bought.
        """
        if amount_kwh == None:
            amount_kwh = self.charger_rate

        amount_bought: float = 0.0

        if self.curr_capacity + amount_kwh <= self.total_capacity*0.8:
            amount_bought += amount_kwh
            self.curr_capacity += amount_kwh
        else: 
            amount_bought += self.total_capacity*0.8 - self.curr_capacity
            self.curr_capacity = self.total_capacity*0.8

        return amount_bought

    def discharge(self, amount_kwh: float = 0.0) -> float:
        """
        Discharges battery and returns the amount of kWh that could not be charged.
        """
        consumption_left: float = 0.0

        if self.curr_capacity - amount_kwh >= self.total_capacity*0.2:
            self.curr_capacity -= amount_kwh
        else:
            consumption_left += amount_kwh - (self.curr_capacity - self.total_capacity*0.2)
            self.curr_capacity = self.total_capacity*0.2

        return consumption_left