import numpy as np
from tire.tire_utils import H_R20_18X6_7
import tire.tire_utils
from tire.tire_model_pacejka_2010 import tire_model_from_arr
import plotly.graph_objs as go
import time
from tire.constants import *
from scipy.optimize import minimize
# from Functions.py_functions.maths import vel_at_tire, clip, to_vel_frame, to_car_frame

def loss_func_two(bd, car, ay_targ, vel, mu_corr, sr_lim):
    ay, yaw, ax, bruh = car.solve_for_yaw(ay_targ, vel, bd[0], bd[1], mu_corr, sr_lim=sr_lim)
    return (ay - ay_targ)**2 + (yaw/5)**2 + ax**2

def variable_sr(v_a, v_b, sr):
    reference_vel = (v_a + v_b) / 2
    ref_slip_speed = reference_vel * (sr + 1)
    return (ref_slip_speed / v_a) - 1, (ref_slip_speed / v_b) - 1

def sr_variable_lim(v_a, v_b, sr, upper, lower):
    reference_vel = (v_a + v_b) / 2
    
    # Derive bounds for sr based on sr_a
    lower_bound_a = (lower + 1) * v_a / reference_vel - 1
    upper_bound_a = (upper + 1) * v_a / reference_vel - 1

    # Derive bounds for sr based on sr_b
    lower_bound_b = (lower + 1) * v_b / reference_vel - 1
    upper_bound_b = (upper + 1) * v_b / reference_vel - 1

    # Determine the overlapping region between the two bounds
    final_lower_bound = max(lower_bound_a, lower_bound_b)
    final_upper_bound = min(upper_bound_a, upper_bound_b)

    # If current sr is within the bounds, return it. Otherwise, return a bound value.
    if final_lower_bound <= sr <= final_upper_bound:
        return sr
    else:
        # Return the bound that's closest to the original sr
        if abs(final_lower_bound - sr) < abs(final_upper_bound - sr):
            return final_lower_bound
        else:
            return final_upper_bound

def vel_at_tire(v, omega, beta, x, y):
    v_x = v * np.cos(beta) + omega * y
    v_y = v * np.sin(beta) + omega * x
    v_v = np.sqrt(v_x**2 + v_y**2)
    return v_v

class Tire:
    def __init__(self, data_arr=H_R20_18X6_7,scaling_factor:float = 0.6) -> None:
        self.set_tire(data_arr)
        self.scaling_factor=scaling_factor
        

    def set_tire(self, tire):
        self.mf_tire = tire_model_from_arr(tire)
        try:
            self.fast_mf = get_rs_pacejka(self.mf_tire)
        except:
            print("Failed to load fast pacejka, using slow pacejka, you should really compile the fast pacejka its like 3x faster")
            self.fast_mf = None

    def s_r_ind_edif(self, f_z_a, s_a_a, i_a_a, v_a, f_z_b, s_a_b, i_a_b, v_b, fx_target, upper = 0.2, lower = -0.3, p: float = 82500, flip_s_a=False, non_driven=False, mu_corr: float = 1.0):
        kappa_a, bam_a, fx_a = self.s_r_sel(f_z_a, s_a_a, i_a_a, v_a, fx_target / 2, non_driven=non_driven, upper=upper, lower=lower, flip_s_a=flip_s_a, mu_corr=mu_corr, p=p)
        fx_b_targ = fx_target
        if bam_a or fx_a == 0:
            fx_b_targ -= fx_a
        kappa_b, bam_b, fx_b = self.s_r_sel(f_z_b, s_a_b, i_a_b, v_b, fx_b_targ, non_driven=non_driven, upper=upper, lower=lower, flip_s_a=(not flip_s_a), mu_corr=mu_corr, p=p)
        # print(f"fx_targ: {fx_target:.4f} fx_a: {fx_a:.4f} fx_b_targ: {fx_b_targ:.4f}, bam_a: {bam_a} kappa_a: {kappa_a:.4f} fx_b: {fx_b:.4f}, bam_b: {bam_b} kappa_b: {kappa_b:.4f}")
        return kappa_a, bam_a, fx_a, kappa_b, bam_b, fx_b
    
    def s_r_ind_locked(self, f_z_a, s_a_a, i_a_a, v_a, f_z_b, s_a_b, i_a_b, v_b, fx_target, upper = 0.2, lower = -0.3, p: float = 82500, non_driven=False, mu_corr: float = 1.0, og_upper = 0.2, og_lower = -0.3, kappa=0.0, prev_kappa=0.0, prev_fx=[0.0, 0.0], i=0):
        """
        Solves for the slip ratio of a locked differential, also determines if the tire is saturated with Fx
        To do this, we use a second order taylor series approximation to solve for the slip ratio
        """
        if (fx_target > 0 and non_driven): # If the tire is non driven (eg front wheels) and the target Fx is positive (acceleration), then the tire is wont be reacting any torque
            _, actual_fx_a, _ = self.steady_state_mmd(f_z_a, s_a_a, 0.0, v_a, i_a_a, 0.0, flip_s_a=True, mu=mu_corr, no_long_include=True, p=p)
            _, actual_fx_b, _ = self.steady_state_mmd(f_z_b, s_a_b, 0.0, v_b, i_a_b, 0.0, flip_s_a=False, mu=mu_corr, no_long_include=True, p=p)
            return 0.0, False, actual_fx_a, 0.0, False, actual_fx_b
        if f_z_a <= 0.0:
            kappa_b, bam_b, fx_b = self.s_r_sel(f_z_b, s_a_b, i_a_b, v_b, fx_target / 2, upper=upper, lower=lower, non_driven=non_driven, mu_corr=mu_corr, flip_s_a=False, p=p)
            return 0.0, False, 0.0, kappa_b, bam_b, fx_b
        if f_z_b <= 0.0:
            kappa_a, bam_a, fx_a = self.s_r_sel(f_z_a, s_a_a, i_a_a, v_a, fx_target / 2, upper=upper, lower=lower, non_driven=non_driven, mu_corr=mu_corr, flip_s_a=True, p=p)
            return kappa_a, bam_a, fx_a, 0.0, False, 0.0
        if i > 20:
            k_a, k_b = variable_sr(v_a, v_b, prev_kappa)
            return k_a, False, prev_fx[0], k_b, False, prev_fx[1]
        # first we solve for 3 points with a small offset of b from our slip ratio kappa to get the first and second derivatives
        # here is what is going on here https://mathformeremortals.wordpress.com/2013/01/12/a-numerical-second-derivative-from-three-points/
        b = 0.0001
        d_kappa = 0.001
        kappas = np.array([kappa - b, kappa, kappa + b])
        k_a, k_b = variable_sr(v_a, v_b, kappas)
        # With a locked diff there is low speed locked diff behavior and so the need to disable some of the error checking in the solver that is valid for a one wheel iterator
        if self.fast_mf == None:
            fx_a, _, _ = self.mf_tire.s_r_sweep(f_z_a, s_a_a, k_a, i_a=i_a_a, v=v_a, flip_s_a=True, mu_corr=mu_corr, p=p)
            fx_b, _, _ = self.mf_tire.s_r_sweep(f_z_b, s_a_b, k_b, i_a=i_a_b, v=v_b, flip_s_a=False, mu_corr=mu_corr, p=p)
        else:
            fx_a, _, _ = self.fast_mf.solve_sr_sweep(f_z_a, s_a_a, k_a, p, i_a_a, v_a, 0.0, 0.0, mu_corr, True)
            fx_b, _, _ = self.fast_mf.solve_sr_sweep(f_z_b, s_a_b, k_b, p, i_a_b, v_b, 0.0, 0.0, mu_corr, False)
        # now we use the first and second derivatives to solve for the slip ratio
        fx = fx_a + fx_b
        fx_1, fx_2, fx_3 = fx[0], fx[1], fx[2]
        d_fx = (fx_3 - fx_1) / (2 * b)
        dd_fx = (fx_3 - 2 * fx_2 + fx_1) / (b ** 2)
        max_fxy_mag = 3 * (f_z_a + f_z_b) # limit the maxima used in the quadratic equation, it can jump all the way off the other end of the curve if we dont
        delta_fx = np.clip(fx_target, -max_fxy_mag, max_fxy_mag) - fx_2
        if d_fx ** 2 - 4 * dd_fx * delta_fx < 0:
            # use linear approximation if the quadratic equation has no real roots
            new_kappa = kappa + delta_fx / d_fx
            kappa_1 = 0.0
            kappa_2 = 0.0
        else:
            kappa_1 = (-d_fx + np.sqrt(d_fx ** 2 - 4 * dd_fx * delta_fx)) / (2 * dd_fx)
            kappa_2 = (-d_fx - np.sqrt(d_fx ** 2 - 4 * dd_fx * delta_fx)) / (2 * dd_fx)
            new_kappa = kappa - kappa_1 if abs(kappa_1) < abs(kappa_2) else kappa - kappa_2
        if d_fx < 0:
            new_kappa = (prev_kappa + kappa) / 2
            if i == 0 and (fx_a[1] < fx_a[0] or fx_a[1] < fx_a[2] or fx_b[1] < fx_b[0] or fx_b[1] < fx_b[2]):
                print(f"{f_z_a:.1f} BAD TIRE MODEL: NEGATIVE FX-SL SLOPE AT SL=0 i:{i}")

        if abs(new_kappa - kappa) < 0.0001 or abs(fx_target - fx_2) < 0.1:
            km_a, km_b = variable_sr(v_a, v_b, new_kappa)
            maxima_a = (km_a > upper - d_kappa) or (km_a < lower + d_kappa) or ((np.sign(fx_a[1] - fx_a[2]) == np.sign(fx_a[1])) and (np.sign(fx_a[1] - fx_a[0]) == np.sign(fx_a[1])))
            maxima_b = (km_b > upper - d_kappa) or (km_b < lower + d_kappa) or ((np.sign(fx_b[1] - fx_b[2]) == np.sign(fx_b[1])) and (np.sign(fx_b[1] - fx_b[0]) == np.sign(fx_b[1])))
            # if maxima: print(f"{f_z:.1f} MAXIMA")
            return km_a, maxima_a, fx_a[1], km_b, maxima_b, fx_b[1]
        if (kappa == og_upper and new_kappa > og_upper) or (kappa == og_lower and new_kappa < og_lower):
            # print(f"{f_z:.1f} LIMS")
            return k_a[1], True, fx_a[1], k_b[1], True, fx_b[1]
        # this is a stupid way of doing this, but it will stay until this code is completely rewritten
        new_kappa = sr_variable_lim(v_a, v_b, new_kappa, upper, lower)
        if d_fx < 0:
            if kappa > 0:
                upper = kappa
            else: # if kappa < 0
                lower = kappa
            kappa = prev_kappa
            fx_2 = prev_fx[0] + prev_fx[1]
        return self.s_r_ind_locked(f_z_a, s_a_a, i_a_a, v_a, f_z_b, s_a_b, i_a_b, v_b, fx_target, i=(i+1), upper = upper, lower = lower, og_upper = og_upper, og_lower = og_lower, kappa=new_kappa, prev_kappa=kappa, prev_fx=[fx_a[1], fx_b[1]], p=p, non_driven=non_driven, mu_corr=mu_corr)

    def s_r_ind(self, f_z_a, s_a_a, i_a_a, v_a, f_z_b, s_a_b, i_a_b, v_b, fx_target, upper = 0.2, lower = -0.3, p: float = 82500, non_driven=False, rear=True, mu_corr: float = 1.0):
        """
        Solves for the slip ratio of two opposing tires, also determines if one of the tires are saturated with Fx
        If one of the tires is saturated, we know that andy more torque applied to the wheel will bring the tire beyond its peak grip
        While in reality there are some cases there may be more performant to apply more torque to the wheel beyond its peak Fx
        to bring the other wheel closer to its peak Fx and saturate the the tires as a pair, we make the simplifying assumption that this method gets close enough
        this is done because it makes applying torque limits per axle simpler
        """
        # if rear: diff_model = self.diff_model_rear
        # else: diff_model = self.diff_model_front
        diff_model = 'locked'
        
        if diff_model == "open":
            kappa_a, bam_a, fx_a = self.s_r_sel(f_z_a, s_a_a, i_a_a, v_a, fx_target / 2, upper=upper, lower=lower, non_driven=non_driven, mu_corr=mu_corr, flip_s_a=True, p=p)
            kappa_b, bam_b, fx_b = self.s_r_sel(f_z_b, s_a_b, i_a_b, v_b, fx_target / 2, upper=upper, lower=lower, non_driven=non_driven, mu_corr=mu_corr, flip_s_a=False, p=p)
            hysteresis = 10
            if fx_a - hysteresis > fx_b:
                kappa_a, bam_a, fx_a = self.s_r_sel(f_z_a, s_a_a, i_a_a, v_a, fx_b, upper=upper, lower=lower, non_driven=non_driven, mu_corr=mu_corr, flip_s_a=True, p=p)
            elif fx_b - hysteresis > fx_a:
                kappa_b, bam_b, fx_b = self.s_r_sel(f_z_b, s_a_b, i_a_b, v_b, fx_a, upper=upper, lower=lower, non_driven=non_driven, mu_corr=mu_corr, flip_s_a=False, p=p)
            # print(f"fx_a: {fx_a:.3f}, fx_b: {fx_b:.3f}, fx_target: {(fx_target/2):.3f} kappa_a: {kappa_a:.3f}, kappa_b: {kappa_b:.3f} bam_a: {bam_a}, bam_b: {bam_b} s_a_a: {np.rad2deg(s_a_a):.3f}, s_a_b: {np.rad2deg(s_a_b):.3f} i_a_a: {np.rad2deg(i_a_a):.3f}, i_a_b: {np.rad2deg(i_a_b):.3f} v_a: {v_a:.3f}, v_b: {v_b:.3f}")
    
        elif diff_model == "locked":
            kappa_a, bam_a, fx_a, kappa_b, bam_b, fx_b = self.s_r_ind_locked(f_z_a, s_a_a, i_a_a, v_a, f_z_b, s_a_b, i_a_b, v_b, fx_target, upper=upper, lower=lower, non_driven=non_driven, mu_corr=mu_corr, p=p)

        else:
            if f_z_a < f_z_b:
                # print("R Heavy")
                kappa_a, bam_a, fx_a, kappa_b, bam_b, fx_b = self.s_r_ind_edif(f_z_a, s_a_a, i_a_a, v_a, f_z_b, s_a_b, i_a_b, v_b, fx_target, upper=upper, lower=lower, non_driven=non_driven, flip_s_a=True, mu_corr=mu_corr, p=p)
            else:
                # print("L Heavy")
                kappa_b, bam_b, fx_b, kappa_a, bam_a, fx_a = self.s_r_ind_edif(f_z_b, s_a_b, i_a_b, v_b, f_z_a, s_a_a, i_a_a, v_a, fx_target, upper=upper, lower=lower, non_driven=non_driven, flip_s_a=False, mu_corr=mu_corr, p=p)
        
        return kappa_a, kappa_b, (bam_a or bam_b)
    
    def s_r_sel(self, f_z, s_a, i_a, v, fx_targ, flip_s_a=False, upper=0.2, lower=-0.3, p: float = 82500, non_driven=False, mu_corr: float = 1.0):
        if self.fast_mf == None:
            return self.s_r(f_z, s_a, v, fx_targ, i_a=i_a, non_driven=non_driven, upper=upper, lower=lower, og_lower=lower, og_upper=upper, flip_s_a=flip_s_a, mu_corr=mu_corr, p=p)
        else:
            # print(f"{f_z}, {s_a}, {upper:.2f}, {lower:.2f}, {upper:.2f}, {lower:.2f}, 0.0, 0.0, 0.0, {p}, {i_a}, {v}, 0.0, 0.0, {mu_corr}, {flip_s_a}, {non_driven}, {fx_targ}, 0")
            kappa_a, bam_a, fx_a = self.fast_mf.s_r(f_z, s_a, upper, lower, upper, lower, 0.0, 0.0, 0.0, p, i_a, v, 0.0, 0.0, mu_corr, flip_s_a, non_driven, fx_targ, 0)
            # print(f"kappa_a: {kappa_a:.3f}, bam_a: {bam_a}, fx_a: {fx_a:.3f}")
            return kappa_a, bam_a, fx_a

    def s_r(self, f_z, s_a, v_avg, fx_target, i_a = 0.0, upper = 0.2, lower = -0.3, og_upper = 0.2, og_lower = -0.3, kappa=0.0, prev_kappa=0.0, prev_fx=0.0, i=0, p: float = 82500, non_driven=False, flip_s_a=False, mu_corr: float = 1.0):
        """
        Solves for the slip ratio of a single tire, also determines if the tire is saturated with Fx
        To do this, we use a second order taylor series approximation to solve for the slip ratio
        """
        if (fx_target > 0 and non_driven): # If the tire is non driven (eg front wheels) and the target Fx is positive (acceleration), then the tire is wont be reacting any torque
            _, actual_fx, _ = self.steady_state_mmd(f_z, s_a, 0.0, v_avg, i_a, 0.0, flip_s_a=flip_s_a, mu=mu_corr, no_long_include=True, p=p)
            return 0.0, False, actual_fx
        if f_z <= 0.0:
            return 0.0, False, 0.0
        if i > 20:
            return prev_kappa, False, prev_fx
        # first we solve for 3 points with a small offset of b from our slip ratio kappa to get the first and second derivatives
        # here is what is going on here https://mathformeremortals.wordpress.com/2013/01/12/a-numerical-second-derivative-from-three-points/
        b = 0.0001
        d_kappa = 0.001
        kappas = np.array([kappa - b, kappa, kappa + b])
        fx, _, _ = self.mf_tire.s_r_sweep(f_z, s_a, kappas, i_a=i_a, v=v_avg, flip_s_a=flip_s_a, mu_corr=mu_corr, p=p)
        # now we use the first and second derivatives to solve for the slip ratio
        fx_1, fx_2, fx_3 = fx[0], fx[1], fx[2]
        d_fx = (fx_3 - fx_1) / (2 * b)
        dd_fx = (fx_3 - 2 * fx_2 + fx_1) / (b ** 2)
        max_fxy_mag = 3 * f_z # limit the maxima used in the quadratic equation, it can jump all the way off the other end of the curve if we dont
        delta_fx = np.clip(fx_target, -max_fxy_mag, max_fxy_mag) - fx_2
        if d_fx ** 2 - 4 * dd_fx * delta_fx < 0:
            # use linear approximation if the quadratic equation has no real roots
            new_kappa = kappa + delta_fx / d_fx
            kappa_1 = 0.0
            kappa_2 = 0.0
        else:
            kappa_1 = (-d_fx + np.sqrt(d_fx ** 2 - 4 * dd_fx * delta_fx)) / (2 * dd_fx)
            kappa_2 = (-d_fx - np.sqrt(d_fx ** 2 - 4 * dd_fx * delta_fx)) / (2 * dd_fx)
            new_kappa = kappa - kappa_1 if abs(kappa_1) < abs(kappa_2) else kappa - kappa_2
        if d_fx < 0:
            new_kappa = (prev_kappa + kappa) / 2
            if i == 0:
                print(f"{f_z:.1f} BAD TIRE MODEL: NEGATIVE FX-SL SLOPE AT SL=0")

        if abs(new_kappa - kappa) < 0.0001 or abs(fx_target - fx_2) < 0.1:
            maxima = (new_kappa > upper - d_kappa) or (new_kappa < lower + d_kappa) or ((np.sign(fx_2 - fx_3) == np.sign(fx_2)) and (np.sign(fx_2 - fx_1) == np.sign(fx_2)))
            # if maxima: print(f"{f_z:.1f} MAXIMA")
            return new_kappa, maxima, fx_2
        if (kappa == og_upper and new_kappa > og_upper) or (kappa == og_lower and new_kappa < og_lower):
            # print(f"{f_z:.1f} LIMS")
            return kappa, True, fx_2
        new_kappa = max(min(new_kappa, upper), lower)
        if d_fx < 0:
            if kappa > 0:
                upper = kappa
            else: # if kappa < 0
                lower = kappa
            kappa = prev_kappa
            fx_2 = prev_fx
        return self.s_r(f_z, s_a, v_avg, fx_target, i_a=i_a, i=(i+1), upper = upper, lower = lower, og_upper = og_upper, og_lower = og_lower, kappa=new_kappa, prev_kappa=kappa, prev_fx=fx_2, p=p, non_driven=non_driven, flip_s_a=flip_s_a, mu_corr=mu_corr)

    def steady_state_mmd(self, fz, sa, kappa, v_avg, i_a, alpha, p: float = 82500, flip_s_a=False, mu: float = 1.0, no_long_include=False):
        if self.fast_mf == None:
            fx, fy, mz = self.mf_tire.steady_state_mmd(fz, sa, kappa, v=v_avg, flip_s_a=flip_s_a, i_a=i_a, mu_corr=mu, p=p)
        else:
            fx, fy, mz = self.fast_mf.solve_steady_state(fz, sa, kappa, p, i_a, v_avg, 0.0, 0.0, mu, flip_s_a)
        # it is in adapted ISO so pos Fy is neg lat acc
        if no_long_include:
            return -fy * np.cos(-alpha), fx * np.cos(-alpha), -mz
        return -fy * np.cos(-alpha) - fx * np.sin(-alpha), fx * np.cos(-alpha) + -fy * np.sin(-alpha), -mz # per the z down orientation of the ttc data

LC0 = Tire(tire.tire_utils.H_LC0_18X6_7)