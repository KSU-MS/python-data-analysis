import math

def calculate_resistance(rho, length, area):
    return rho * (length / area)

def calculate_temperature_change(current, resistance, time, mass, specific_heat_capacity, h, area, ambient_temperature):
    dt_heating = (current**2 * resistance * time) / (mass * specific_heat_capacity)
    dt_cooling = -h * area * (ambient_temperature - (mass / (8.96e3 * area))) / (mass * specific_heat_capacity)
    return dt_heating + dt_cooling

# Constants for copper
rho_copper = 1.68e-8  # Ohm-meter
specific_heat_capacity_copper = 385  # J/kg°C
h_air = 1  # Heat transfer coefficient (W/m²°C)

# Inputs
current = 3.3  # Amperes
length = 3  # meters
radius = (1.19/1000)/2
radius = ((.97/1000)/2)
wires = {
    "18awg": 3.14*(((1.19/1000)/2)**2),
    "20awg": 3.14*(((0.97/1000)/2)**2),
    "22awg": 3.14*(((0.76/1000)/2)**2),
    "24awg": 3.14*(((0.61/1000)/2)**2)
}
area = 3.14*(radius**2)  # square meters (1 mm^2)
print(area)
time = 24*60  # seconds (24mins * 60 seconds = endurance)
mass = 8.96e3 * length * area  # kg (density of copper * volume)
ambient_temperature = 30  # °C

# Calculating resistance
resistance = calculate_resistance(rho_copper, length, area)
print(f"Resistance: {resistance}ohms")
# Calculating temperature change
print(resistance)
temperature_change = calculate_temperature_change(current, resistance, time, mass, specific_heat_capacity_copper, h_air, area, ambient_temperature)
absolute_max_temp = 75 # max temp period
max_temp_rise = absolute_max_temp - ambient_temperature
for gauge,area in wires.items():
    resistance = calculate_resistance(rho_copper,length,area)
    print(f"wire: {gauge}, area: {area*(1000**2):.4f}mm^2, resistance: {resistance*1000:.3f}mOhms")
    current = 0
    while (calculate_temperature_change(current, resistance, time, mass, specific_heat_capacity_copper, h_air, area, ambient_temperature) < max_temp_rise):
        current+=0.001
    temperature_change = calculate_temperature_change(current, resistance, time, mass, specific_heat_capacity_copper, h_air, area, ambient_temperature)
    print(f"temp rise: {temperature_change:.3f}degC @ {current:.3f}A")