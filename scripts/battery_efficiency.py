# battery degradation object
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


class BatteryEfficiency():
	def __init__(self, battery_capacity_kwh):
		self.battery_capacity = battery_capacity_kwh * 1000 # in Wh
		self._params = { # (Kim & Qiao, 2011) : https://digitalcommons.unl.edu/cgi/viewcontent.cgi?article=1210&context=electricalengineeringfacpub
			'a_0': -0.852, 	'a_1': 63.867, 	'a_2': 3.6297, 	'a_3': 0.559,
			'a_4': 0.51, 	'a_5': 0.508, 	'b_0': 0.1463, 	'b_1': 30.27,
			'b_2': 0.1037, 	'b_3': 0.0584, 	'b_4': 0.1747, 	'b_5': 0.1288,
			'c_0': 0.1063, 	'c_1': 62.94, 	'c_2': 0.0437, 	'd_0': -200,
			'd_1': -138, 	'd_2': 300,		'e_0': 0.0712, 	'e_1': 61.4,
			'e_2': 0.0288, 	'f_0': -3083, 	'f_1': 180, 	'f_2': 5088,
			'y1_0': 2863.3, 'y2_0': 232.66, 'c': 0.9248, 	'k': 0.0008
			}
		self._ref_volts = 3.6 # corrected to the given parameters of morstyn - # check cao if methodology changed, previous: self._ref_volts = 4.2
		self._cellnum = int(np.ceil(self.battery_capacity / self._ref_volts))
		# self._cellnum = 1

	def ss_circuit_model(self, soc):
		v_oc = ((self._params['a_0'] * np.exp(-self._params['a_1'] * soc)) + self._params['a_2'] + (self._params['a_3'] * soc) - (self._params['a_4'] * soc**2) + (self._params['a_5'] * soc**3)) * self._cellnum
		r_s = ((self._params['b_0'] * np.exp(-self._params['b_1'] * soc)) + self._params['b_2'] + (self._params['b_3'] * soc) - (self._params['b_4'] * soc**2) + (self._params['b_5'] * soc**3)) * self._cellnum
		r_st = (self._params['c_0'] * np.exp(-self._params['c_1'] * soc) + self._params['c_2']) * self._cellnum
		r_tl = (self._params['e_0'] * np.exp(-self._params['e_1'] * soc) + self._params['e_2']) * self._cellnum # parameters in Cao look suspecious, the one in "Accurate Electrical Battery -
		# Model Capable of Predicting Runtime and I–V Performance" closer to e_1 - e_3

		r_tot = (r_s + r_st + r_tl)
		return v_oc, r_tot, r_s, r_st, r_tl

	def circuit_current(self, v_oc, r_tot, p_e):
		icur = (v_oc - np.sqrt(v_oc**2 - 4 * r_tot * p_e)) / (2 * r_tot)
		return icur

	def calc_efficiency(self, v_oc, r_tot, icur, p_e):  # p_r > 0 is discharging
		if p_e < 0:   # charging
			efficiency =  v_oc / (v_oc - (r_tot * icur)) # past auch nach morstyn
		elif p_e > 0: # discharging
			efficiency =  (v_oc - (r_tot * icur)) / v_oc # past auch nach morstyn
		else:
			efficiency = 1.0 

		return efficiency



	def calc_efficiency_all(self, current_soc, p):
		p_e = p * 1000 # in W
		v_oc, r_tot, r_s, r_st, r_tl = self.ss_circuit_model(current_soc)
		icur = self.circuit_current(v_oc, r_tot, p_e)
		efficiency = self.calc_efficiency(v_oc, r_tot, icur, p_e)
		return efficiency