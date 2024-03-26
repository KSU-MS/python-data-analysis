import numpy as np
import cmath
import math 
import pandas as pd
import warnings
warnings.simplefilter(action='ignore',category=pd.errors.PerformanceWarning)
warnings.simplefilter(action='ignore',category=pd.errors.SettingWithCopyWarning)

class Motor:
    def __init__(self, name):
        self.name = name
        self.max_rate = 6000 # rpm
        self.max_torque = 231 # Nm
        self.max_phase_current = 453 # A
        self.I_dmax = 221 # A
        self.I_qmax = 453 # A
        self.Ld = 0.000096 # H
        self.Ld_gain = 0.000000 # H/A
        self.Lq = 0.000099 # H
        self.Lq_gain = 0.000000 # H/A
        self.Rs = 0.0071 # Ohm
        self.poles = 10
        self.Fl = 0.03737 # Wb
        
    def __str__(self) -> str:
        return f"{self.name} {self.max_rate} {self.max_torque} {self.max_phase_current} Id max: {self.I_dmax} Iq max: {self.I_qmax} Ld: {self.Ld} gain: {self.Ld_gain} Lq: {self.Lq} gain: {self.Lq_gain} Rs: {self.Rs} Poles: {self.poles} Flux: {self.Fl}"
    
    def get_torque(self, I_d, I_q):
        Ld, Lq = self.get_inductance(I_d, I_q)
        return 3/2*self.poles*(self.Fl*I_q + (Ld-Lq)*I_d*I_q)
        return 3/2 * self.poles * I_q * (self.Fl + I_d * (Ld - Lq))
        # return 1.1 *(I_q / np.sqrt(2))
    def get_inductance(self, I_d, I_q):
        return self.Ld + I_d * self.Ld_gain, self.Lq + I_d * self.Lq_gain # get the change in inductance due to Id current
    
    def get_qd_currents(self, speed, torque_req, voltage, use_mtpa=False,powerlim=None):
        if powerlim and speed>0:
            # print(f"powerlimited: {torque_req} {(powerlim*9.5488/speed)}")
            torque_req=min((powerlim*9.5488/speed),torque_req)
        # print(f"Speed: {speed} Torque req: {torque_req} Voltage: {voltage}")
        w = speed * 2 * np.pi / 60
        w_e = w * self.poles
        v_max = voltage / np.sqrt(3) - self.Rs * self.max_phase_current
        # https://www.mathworks.com/help/mcb/ref/mtpacontrolreference.html
        I_m_ref = 2 * torque_req / (3 * self.poles * self.Fl)
        I_m = min(I_m_ref, self.max_phase_current)
        if use_mtpa:
            I_d_mtpa = (self.Fl/(4 * (self.Lq - self.Ld))) - np.sqrt((self.Fl**2/(16 * (self.Lq - self.Ld)**2)) + I_m**2/2)
            I_q_mtpa = np.sqrt(I_m**2 - I_d_mtpa**2)
            if ('208' in self.name):
                print(f"208 IQ: {I_q_mtpa}")
        else:
            I_q_mtpa = torque_req * 2 / (3 * self.poles * self.Fl)
            I_d_mtpa = 0
        w_base = (1/self.poles) * (v_max / np.sqrt((self.Lq * I_q_mtpa)**2 + (self.Fl + self.Ld * I_d_mtpa)**2))
        I_q_mtpa = min(I_q_mtpa,self.I_qmax)
        # get the smallest root that is greater than 0
        # Binary search for max T_ref with real solutions
        T_ref_low = 0  # you may adjust this based on any given bounds
        T_ref_high = torque_req * 2.0  # this will make the middle value the first guess
        T_ref_mid = (T_ref_low + T_ref_high) / 2.0
        epsilon = 1e-5  # precision
        last_valid = 0
        was_good = False
        while T_ref_high - T_ref_low > epsilon:
            T_ref_mid = (T_ref_low + T_ref_high) / 2.0
            if any(np.isreal(self.calc_iq_fw_roots(w_e, v_max, T_ref_mid))):
                last_valid = T_ref_mid
                was_good = True
                if T_ref_low == 0 and T_ref_high == torque_req * 2.0:
                    break
                T_ref_low = T_ref_mid
            else:
                T_ref_high = T_ref_mid
                was_good = False
        if not was_good:
            T_ref_mid = last_valid

        roots = self.calc_iq_fw_roots(w_e, v_max, T_ref_mid)
        roots = roots[np.isreal(roots)]
        I_q_fw = min(max(max(roots.real), 0), self.I_qmax)
        I_d_fw = -self.Fl / self.Ld + (1/self.Ld) * np.sqrt((v_max**2/w_e**2)-(self.Lq*I_q_fw)**2)
        if I_d_fw > 0:
            I_d_fw = 0
        if I_d_fw < -self.I_dmax:
            I_d_fw = -self.I_dmax
            # print(f"Funky iq: {I_q_fw}")
            I_q_fw = np.sqrt((v_max**2/w_e**2)-(self.Ld*I_d_fw+self.Fl)**2)/self.Lq # self.Fl + 
        # I_d_fw = min(max(I_d_fw, -self.max_phase_current), 0)
        # print(f"FW IQ: {I_q_fw} ID: {I_d_fw}")
        # if I_q_fw >= I_q_mtpa:
        #     I_q_fw=I_q_mtpa
        # if I_q_fw <= I_q_mtpa:
        #     I_q_fw=I_q_mtpa
        if np.isnan(I_q_fw):
            I_q_fw=0
        if w <= w_base:
            I_d = I_d_mtpa
            I_q = I_q_mtpa
        else:
            I_d = I_d_fw
            I_q = I_q_fw
        if math.sqrt(I_d**2 + I_q**2) > (self.max_phase_current):
            I_q = math.sqrt(self.max_phase_current**2 - I_d**2)
        achieved_torque = self.get_torque(I_d, I_q)
        # print(f"IQ: {I_q} ID: {I_d}")
        if achieved_torque > torque_req: achieved_torque = torque_req
        return I_d, I_q, achieved_torque, w_base, v_max
    
    def calc_iq_fw_roots(self, w_e, v_max, torque_req):
        # Long ahh equation from here https://www.mathworks.com/help/mcb/ref/mtpacontrolreference.html
        coeffs = [
            9 * self.poles**2 * (self.Ld - self.Lq)**2 * self.Lq**2 * w_e**2,  # Coefficient of i_{q_fw}^4
            0,  # Coefficient of i_{q_fw}^3 (since it doesn't appear in your equation)
            9 * self.poles**2 * self.Fl**2 * self.Lq**2 * w_e**2 - 9 * self.poles**2 * (self.Ld - self.Lq)**2 * v_max**2,  # Coefficient of i_{q_fw}^2
            -12 * torque_req * self.poles * self.Fl * self.Ld * self.Lq * w_e**2,  # Coefficient of i_{q_fw}
            4 * torque_req**2 * self.Ld**2 * w_e**2  # Constant term
        ]

        # Calculate the roots using NumPy
        return np.roots(coeffs)

    @property
    def i_max(self):
        return self.max_phase_current * np.sqrt(2)

    def calc_mtpa(self, current):
        i_dmtpa = (self.Fl - np.sqrt(self.Fl**2 - 8 * ((self.Lq - self.Ld)**2) * current**2)) / (4 * (self.Ld - self.Lq))
        i_qmtpa = np.sqrt(i_dmtpa**2 - ((self.Fl / (self.Lq - self.Ld)) * i_dmtpa))
        return i_dmtpa, i_qmtpa
    
    def calc_iq(self, torque, id):
        return torque / ((3/2) * self.poles * (self.Fl - ((self.Lq - self.Ld) * id)))
    
    def calc_within_emf(self, speed, voltage, id, iq):
        w = speed * 2 * np.pi / 60
        u_ac = voltage / np.sqrt(2)
        v_d = self.Rs * id - w * self.Lq * iq
        v_q = self.Rs * iq + w * (self.Fl + self.Ld * id)
        v_ac = np.sqrt(v_d**2 + v_q**2)
        return v_ac <= u_ac
    
    def calc_emf_limit(self, speed, voltage, points):
        w = self.poles * speed * 2 * np.pi / 60
        u_ac = voltage / np.sqrt(2)
        # Calculate points along the profile where v_ac = u_ac, we will solve for points at different angles from the center of the limit profile
        # The center of the limit profile is at the point (0, -l_d_center)
        l_d_center = -self.Fl / self.Ld
        angles = np.linspace(-np.pi, np.pi, points)
        m = np.tan(angles)
        s = -l_d_center * m
        # Solve for the points where the limit profile intersects the solution lines
        # i_q = m * i_d + b
        a = self.Rs**2 * (1 + m**2) + w**2 * (self.Ld**2 + (self.Lq**2 * m**2)) + (2 * (self.Ld - self.Lq) * m * self.Rs * w)
        b = (2 * self.Rs * w * (((self.Ld - self.Lq) * s) + (self.Fl * m))) + (2 * w**2 * ((m * s * self.Lq**2) + (self.Fl * self.Ld))) + (2 * m * s * self.Rs**2)
        c = (w**2 * ((self.Lq**2 * s**2) + self.Fl**2)) + (s**2 * self.Rs**2) + (2 * w * self.Fl * self.Rs * s) - u_ac**2
        i_d_neg = (-b - np.sqrt(b**2 - 4 * a * c)) / (2 * a)
        i_d_pos = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)
        i_d = np.zeros(points)
        i_d = i_d_pos
        i_d[angles > np.pi/2] = i_d_neg[angles > np.pi/2]
        i_d[angles < -np.pi/2] = i_d_neg[angles < -np.pi/2]
        i_q = m * i_d + s
        return i_d, i_q
    
    def get_voltages(self, w, id, iq):
        # w = speed * 2 * np.pi / 60
        v_d = self.Rs * id - w * self.Lq * iq
        v_q = self.Rs * iq + w * (self.Fl + self.Ld * id)
        return v_d, v_q

    def calc_current_limit(self, points):
        angles = np.linspace(0, 2*np.pi, points)
        i_d = self.i_max * np.cos(angles)
        i_q = self.i_max * np.sin(angles)
        return i_d, i_q
    
    def power_limit(self, speed, voltage):
        power_limit = 80000
        w = self.poles * speed * 2 * np.pi / 60
        u_ac = voltage / np.sqrt(2)
        peak_iq = power_limit / u_ac # The peak Iq that can be generated with the power limit and current voltage
        id_torque = (3/2) * self.poles * self.Fl * peak_iq # The peak torque that can be produced assuming no field weakening (not mtpa)
        # now we generate the Iq current when the field weakening circle intersects with the Id axis
        a = w^2 * self.Ld^2 + self.Rs^2
        b = 2 * w * self.Fl * self.Rs
        c = self.Fl^2 * w^2 - u_ac^2
        # determine if it even intersects
        if b^2 - 4 * a * c < 0:
            id_fw = 0 # it will be less than the peak_iq and that is all we need to know
        else:
            id_fw = (-b + np.sqrt(b^2 - 4 * a * c)) / (2 * a)
        # now we can look for the intersection of the field weakening circle and the current limit circle
    
EMRAX228MV = Motor('228MV')
EMRAX228MV.Ld = 0.000096
EMRAX228MV.Lq = 0.000099
EMRAX228MV.Rs = 0.0076 #0.0071
EMRAX228MV.max_phase_current = 360.0 * math.sqrt(2)
EMRAX228MV.max_rate = 6500.0
EMRAX228MV.Fl = 0.03737
EMRAX228MV.poles = 10.0
EMRAX228MV.I_dmax = 221 # A
EMRAX228MV.I_qmax = 453 # A

EMRAX228HV = Motor('228HV')
EMRAX228HV.Ld = 0.000177
EMRAX228HV.Lq = 0.000183
EMRAX228HV.Rs = 0.018 #0.0071
EMRAX228HV.max_phase_current = 240.0 * math.sqrt(2)
EMRAX228HV.max_rate = 6500.0
EMRAX228HV.Fl = 0.0542
EMRAX228HV.poles = 10.0
EMRAX228HV.I_dmax = 150 # A
EMRAX228HV.I_qmax = 339 # A
current_lims = 600
torque = 240
speed = 6000.0
voltage = 300

# Emrax 208HV and MV defs
EMRAX208HV = Motor('208HV')
EMRAX208HV.Ld = 0.0001755
EMRAX208HV.Lq = 0.0001760 #  guess
EMRAX208HV.Rs = (12.27)/1000 #milliohms
EMRAX208HV.max_phase_current = 240 * math.sqrt(2)
EMRAX208HV.max_rate = 7000
EMRAX208HV.Fl = 0.03758
EMRAX208HV.poles = 10
EMRAX208HV.I_dmax = 150 # From cascadia setup
EMRAX208HV.I_qmax = 283 # From cascadia setup

EMRAX208MV = Motor('208MV')
EMRAX208MV.Ld = 0.0000735
EMRAX208MV.Lq = 0.0000750 #  guess
EMRAX208MV.Rs = (5.51)/1000 #milliohms
EMRAX208MV.max_phase_current = 400 * math.sqrt(2)
EMRAX208MV.max_rate = 7000
EMRAX208MV.Fl = 0.02338
EMRAX208MV.poles = 10
EMRAX208MV.I_dmax = 425 # From cascadia setup, probably wrong
EMRAX208MV.I_qmax = 425 # From cascadia setup (not wrong)

EMRAX_MOTORS = [EMRAX228MV,EMRAX228HV,EMRAX208MV,EMRAX208HV] # 228mv,228hv,208mv, 208hv

def calcc(w, t, v, motor:Motor):
    Id, Iq, T, wbase, v_max = motor.get_qd_currents(w, t, v)
    return Id, Iq, T, wbase * 60 / (2 * np.pi), v_max

def pdcalcc(row,rpmkey,torquekey,voltagekey,motor:Motor):
    id,iq,torque,speed,vmax = calcc(row[rpmkey],row[torquekey],row[voltagekey],motor=motor)
    return pd.Series([id,iq,torque,speed,vmax])
def generate_power_curve(motor:Motor,maxtorque:int,maxrpm:int,rpmincrement:int,voltage:int):
    current_motor = motor
    print(motor)
    df = pd.DataFrame()

    df['w'] = range(1, maxrpm, rpmincrement)
    df['t'] = maxtorque
    df['v'] = voltage
    theoretical_id = 'id_t'
    theoretical_iq = 'iq_t'
    theoretical_torque = 't_t'
    theoretical_speed = 'w_t'
    theoretical_vmax= 'v_max'
    df[[theoretical_id,theoretical_iq,theoretical_torque,theoretical_speed,theoretical_vmax]] = df.apply(pdcalcc,rpmkey='w',torquekey='t',voltagekey='v',motor=current_motor,axis=1)
    df["power"] = df[theoretical_torque] * df['w'] / 9.5488
    return df

def generate_all_motors(voltage:float=302.4):
    data_list = {
    "emrax208mvtorquecurve":generate_power_curve(EMRAX208MV,160,7000,10,voltage),
    "emrax208hvtorquecurve":generate_power_curve(EMRAX208HV,160,7000,10,voltage),
    "emrax228mvtorquecurve":generate_power_curve(EMRAX228MV,230,6500,10,voltage),
    "emrax228hvtorquecurve":generate_power_curve(EMRAX228HV,230,6500,10,voltage)
    }
    return data_list
def main():
    data_list = generate_all_motors()
    for key,value in data_list.items():
        value.to_csv(key+".csv")
        
if __name__ == "__main__":
    main()