class Battery:
    """
    Simulated battery
    """
    def __init__(self, 
                 battery_total_capacity: float, 
                 battery_curr_capacity: float = 0.0,
                 charger_rate: float = 7.2,
                 discharge_rate: float|None = None,
                 min_soc: float = 0.20,
                 max_soc: float = 0.80,
                 eff_charge: float = 0.95,
                 eff_discharge: float = 0.96):
        
        # Validations
        if battery_total_capacity <= 0.0:
            raise ValueError("total_capacity must be > 0")
        if battery_curr_capacity <= 0.0 and battery_curr_capacity > battery_total_capacity:
            raise ValueError("SoC must be > 0 and < total_capacity")
        if not (0.0 <= min_soc < max_soc <= 1.0):
            raise ValueError("SoC window must satisfy 0 ≤ min_soc < max_soc ≤ 1")
        
        # Capacity
        self.curr_capacity: float  = battery_curr_capacity
        self.total_capacity: float = battery_total_capacity
        self.min_soc: float        = min_soc
        self.max_soc: float        = max_soc

        # Charge and discharge rate
        if discharge_rate is None:
            discharge_rate = self.total_capacity / 2.0
        self.charger_rate: float   = min(charger_rate, self.total_capacity/8)
        self.discharge_rate: float = max(0.0, discharge_rate)

        # Efficiency
        self.eff_charge: float    = max(0.0, min(1.0, eff_charge))
        self.eff_discharge: float = max(0.0, min(1.0, eff_discharge))

        
    def charge(self, amount_kwh: float|None = None) -> float:
        """
        Charges battery and returns the amount of kWh bought.
        """
        if amount_kwh is None:
            amount_kwh = self.charger_rate

        # Limit
        usable_max: float         = self.total_capacity * self.max_soc
        allowed: float            = max(0.0, usable_max - self.curr_capacity)
        max_bought_allowed: float = allowed / max(self.eff_charge, 1e-9) # We buy more because we can't store 100%
        bought: float             = min(amount_kwh, allowed, max_bought_allowed)

        # Update
        self.curr_capacity += bought * self.eff_charge

        return bought

    def discharge(self, amount_kwh: float = 0.0) -> float:
        """
        Discharges battery and returns the amount of kWh that could not be charged.
        """
        # Limit
        usable_min: float = self.total_capacity * self.min_soc
        available: float  = max(0.0, self.curr_capacity - usable_min)

        # How much can we deliver
        max_deliverable_by_energy: float = available * self.eff_discharge
        max_deliverable_by_rate: float   = self.discharge_rate * self.eff_discharge
        can_deliver: float               = min(amount_kwh, max_deliverable_by_rate, max_deliverable_by_energy)
        served: float                    = can_deliver #/ max(self.eff_discharge, 1e-9)

        # Update
        self.curr_capacity     -= served
        consumption_left: float = amount_kwh - served

        return consumption_left