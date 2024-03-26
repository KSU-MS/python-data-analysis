import numpy as np
import pandas as pd

class battery:
    def __init__(self,capacity_voltage_curve_path:str,capacity=2500,resistance=32.1/1000,c_rate = 14) -> None:
        """create an instance of a single battery cell

        Args:
            capacity_voltage_curve_path (str): filepath to csv that describes capacity to voltage relationship
            capacity (int): capacity in mah
            resistance (float): resistance in ohms
        """
        self.maxv = 4.2 # full capacity
        self.nomv = 3.6
        self.minv = 2.0 # no capacity
        self.voltage = 4.2 # start at full charge
        self.maxcapacity = capacity # keep track of the full charge capacity
        self.capacity = capacity # milliamp-hours
        self.maxdischarge = c_rate * capacity/1000 # peak current in amps
        self.resistance =  resistance #internal resistance in ohms
        batteryData = pd.read_csv(capacity_voltage_curve_path)
        coeffs = np.polyfit(batteryData['Ah'],batteryData['Voltage'],8)
        self.voltageUpdateFunction = np.poly1d(coeffs)
        
    def updateVoltage(self):
        if self.capacity>0:
            self.voltage = self.voltageUpdateFunction(self.capacity)
        else:
            self.voltage = 2.5
        # print(f"Cell Voltage: {self.voltage}")
        return self.voltage
    
    def discharge(self,current,time):
        # Calculate the amount of charge discharged (Q = I * t)
        current = current * 1000 # scale amps to milliamps
        time = time / 3600 # scale seconds to hour
        # print(f"current: {current} time: {time}")
        
        discharged_charge = current * time # maH
        
        # print(f"discharged_charge: {discharged_charge}mAH")
        # Calculate the power dissipated as heat (P = I^2 * R)
        heat_power = current**2 * self.resistance # milliwatts
        # print(f"i2r: {heat_power}mW")

        # Convert the power to an equivalent loss of capacity based on the cell's voltage (P = V * I)
        # Assuming average voltage during discharge is half of max voltage
        heat_loss_energy = heat_power * time #should be mah?
        # print(f"heat_loss_energy: {heat_loss_energy}mWh")
        heat_loss_capacity = heat_loss_energy / self.voltage
        # print(f"heat_loss_capacity: {heat_loss_capacity}maH")
        # Adjust the discharged charge for the heat loss
        heat_loss_capacity = 0 # fuck this
        effective_discharged_charge = discharged_charge + heat_loss_capacity
        # print(f"effective discharged: {effective_discharged_charge}")
        # Calculate the new capacity after discharge
        # print(f"capacityBefore: {self.capacity}")
        self.capacity -= effective_discharged_charge
        
        # print(f"capacity now: {self.capacity}")

        # Ensure the capacity does not go below zero
        self.capacity = max(self.capacity, 0)  
        
    def getInstantaneousVoltage(self,current):
        voltage_sag = current * self.resistance
        return self.voltage - voltage_sag
        
class batteryPack:
    def __init__(self,battery:battery,parallelCount:int,seriesCount:int,name:str = None) -> None:
        self.cell = battery
        self.parallelCount = parallelCount
        self.seriesCount = seriesCount
        self.cellCount = self.parallelCount * self.seriesCount
        self.voltage = battery.voltage * self.seriesCount
        self.capacity = battery.capacity * self.parallelCount
        self.maxdischarge = battery.maxdischarge * self.parallelCount
        self.name = name
        
    def __str__(self): 
        return f"{self.name} {self.seriesCount}S {self.parallelCount}P, {self.parallelCount * self.cell.maxcapacity/1000}Ah, Max {self.cell.maxv * self.seriesCount:.2f}V, Nominal {self.cell.nomv * self.seriesCount}V"   
    
    def updateVoltage(self):
        self.cell.updateVoltage()
        self.voltage = self.cell.voltage * self.seriesCount
        
    def discharge(self,current,time):
        self.updateVoltage()
        # print(f"Capacity before: {self.capacity}")
        self.cell.discharge(current/self.parallelCount,time)
        self.capacity = self.cell.capacity * self.parallelCount
        # print(f"Capacity after discharging: {self.capacity}")
        return self.capacity
    
    def getInstantaneousVoltage(self,current):
        self.updateVoltage()
        sagged_cell_v = self.cell.getInstantaneousVoltage(current/self.parallelCount)
        return sagged_cell_v * self.seriesCount
    
    def getMaxPowerOut(self):
        current_limit = self.maxdischarge
        real_max = current_limit * (self.getInstantaneousVoltage(current_limit))
        pack_resistance = self.cell.resistance / self.parallelCount
        pack_resistance *= self.seriesCount
        theoretical_max = (self.getInstantaneousVoltage(0)**2)/(4*pack_resistance)
        return real_max
class cell_models:
    lgHE2= battery(r'battery_soc_curves/lgHE2ahCurve.csv')
    # THis curve is not real, I just adjusted the points of the HE2 curve
    cosmx= battery(r'battery_soc_curves/COSMXahCurve.csv',capacity=13000,resistance=1.5/1000,c_rate=25)
class batt_models:
    energus250v = batteryPack(cell_models.lgHE2,8,60,'ENERGUS60s8p')
    energus300v = batteryPack(cell_models.lgHE2,8,72,'ENERGUS72s8p')
    cosmx300v = batteryPack(cell_models.cosmx,2,72,'COSMX72s2p')
    cosmx400v = batteryPack(cell_models.cosmx,1,96,'COSMX96s1p')
    cosmx600v= batteryPack(cell_models.cosmx,1,144,'COSMX144s1p')