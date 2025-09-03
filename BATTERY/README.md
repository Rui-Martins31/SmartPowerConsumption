## Features to implement

- Simple tapering near max SoC (charge) and protection near min SoC (discharge)
- Add self_discharge_per_day and a tick(dt_hours=1.0) method to be called inside each loop iteration.
- Add SOH (State of Health).
- Add simple temperature scalar and derate charge/discharge rates when cold/hot.