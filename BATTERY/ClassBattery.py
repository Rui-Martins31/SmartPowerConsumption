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
        
    def charge(self, amount_kwh: float = None) -> None:
        if amount_kwh == None:
            amount_kwh = self.charger_rate

        if self.curr_capacity + amount_kwh <= self.total_capacity*0.8:
            self.curr_capacity += amount_kwh
        else: 
            self.curr_capacity = self.total_capacity*0.8

    def discharge(self, amount_kwh: float = 0.0) -> None:
        if self.curr_capacity - amount_kwh >= self.total_capacity*0.2:
            self.curr_capacity -= amount_kwh
        else:
            self.curr_capacity = self.total_capacity*0.2