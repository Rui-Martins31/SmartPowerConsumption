class Battery:
    """
    Simulated battery
    """
    def __init__(self, battery_total_capacity: float, battery_curr_capacity: float = 0.0):
        self.curr_capacity: float = battery_curr_capacity
        self.total_capacity: float = battery_total_capacity


        if self.total_capacity <= 0.0:
            raise ValueError
        if self.curr_capacity <= 0.0:
            raise ValueError
        
    def charge(self, amount_kwh: float) -> None:
        if self.curr_capacity + amount_kwh <= self.total_capacity:
            self.curr_capacity += amount_kwh

    def discharge(self, amount_kwh: float) -> None:
        if self.curr_capacity - amount_kwh >= 0.0:
            self.curr_capacity -= amount_kwh