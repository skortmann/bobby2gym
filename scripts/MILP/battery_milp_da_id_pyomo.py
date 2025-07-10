import pandas as pd
import numpy as np
from pyomo.environ import *
import matplotlib.pyplot as plt
import math
import time

from battery_efficiency import BatteryEfficiency
from battery_degradation_func import calculate_degradation
import itertools


class BatteryMILP():

	def __init__(self, battery_power, battery_capacity):
		# self.df = optimise_df

		self.cap = battery_capacity
		self.pwr = battery_power

		self.previous_cap = 100
		# self.batt_cost = 75000 # £/MWh
		self.kwh_cost = 127
		self.batt_cost = self.kwh_cost * self.cap

		self.hour_resolution = 4

	def optimise(self, price_df, intial_soc, current_cycle_num, remaining_capacity, previous_ep_power):

		"""
		optimise_df : pandas dataframe containing the hourly price with date range as index
		"""


		prices_da = np.array(price_df.price_da_qh.tolist())
		prices_id = np.array(price_df.price_id.tolist())

		# define problem as pyomo's 'concrete' model
		model = ConcreteModel()

		# set params
		# model.T = Set(doc="hour in simulation", ordered=True, initialize=hourly_refs)
		model.T = RangeSet(0, len(price_df.price_id.tolist())-1)
		model.pwr = Param(initialize=self.pwr, doc="power rating of battery (MW)")
		model.cap = Param(initialize=self.cap, doc="capacity rating of battery (MWh)")
		# model.price = Param(model.T, initialize=prices, doc="hourly price (€/MWh)")
		model.price_da = Param(model.T, initialize=prices_da, doc="qh price da(€/MWh)")
		model.price_id = Param(model.T, initialize=prices_id, doc="qh price id(€/MWh)")

		# set charge and discharge varaibles
		model.energy_in = Var(model.T, domain=pyomo.core.NonNegativeReals, initialize=0)
		model.energy_out = Var(model.T, domain=pyomo.core.NonNegativeReals, initialize=0)
		model.energy_in_id = Var(model.T, domain=pyomo.core.NonNegativeReals, initialize=0)
		model.energy_out_id = Var(model.T, domain=pyomo.core.NonNegativeReals, initialize=0)
		model.energy_in_da = Var(model.T, domain=pyomo.core.NonNegativeReals, initialize=0)
		model.energy_out_da = Var(model.T, domain=pyomo.core.NonNegativeReals, initialize=0)


		# set state-of-charge bounds
		model.soc = Var(model.T, bounds=(0.2*model.cap*(remaining_capacity/100), (1*model.cap*(remaining_capacity/100))), initialize=0.5*model.cap*(remaining_capacity/100))

		# set boolean charge and discharge vars
		model.charge_bool= Var(model.T, within=pyomo.core.Boolean, initialize=0)
		model.discharge_bool = Var(model.T, within=pyomo.core.Boolean, initialize=0)

		# store profit on timeseries resolution
		model.profit_timeseries = Var(model.T, within=pyomo.core.Reals, initialize=0)
		model.cumulative_profit = Var(model.T, within=pyomo.core.Reals, initialize=0)

		# declare var for cycle rate
		model.cumlative_cycle_rate = Var(model.T, within=pyomo.core.Reals, initialize=0)

		charging_efficiency = 1.0
		discharge_efficiency = 1.0

		# state of charge constraint
		def update_soc(model, t):
			if t == 0:
				return model.soc[t] == intial_soc - ((model.energy_out_da[t]) + (model.energy_out_id[t])) + (((model.energy_in_da[t]) + (model.energy_in_id[t])))
			else:
				return model.soc[t] == model.soc[t-1] - ((model.energy_out_da[t]) + (model.energy_out_id[t])) + (((model.energy_in_da[t]) + (model.energy_in_id[t])))

		model.state_of_charge = Constraint(model.T, rule=update_soc)

		### ---- added lines to make differertiation between id and da market
		def sum_energy_in(model, t):
			return model.energy_in_id[t] + model.energy_in_da[t] - model.energy_out_id[t] - model.energy_out_da[t] <= model.energy_in[t]

		model.sum_energy_in = Constraint(model.T, rule=sum_energy_in)

		def sum_energy_out(model, t):
			return model.energy_out_id[t] + model.energy_out_da[t] - model.energy_in_id[t] - model.energy_in_da[t] <= model.energy_out[t]

		model.sum_energy_out = Constraint(model.T, rule=sum_energy_out)

		# current charge power constraint
		def charge_constraint(model, t):
			return model.energy_in[t] * self.hour_resolution <= (model.pwr)

		model.charge = Constraint(model.T, rule=charge_constraint)

		# current charge power constraint
		def discharge_constraint(model, t):
			return model.energy_out[t] * self.hour_resolution <= (model.pwr)

		model.discharge = Constraint(model.T, rule=discharge_constraint)

		# Second charge, discharge rule
		def charge_constraint_da(model, t):
			return model.energy_in_da[t] * self.hour_resolution <= (model.pwr)

		model.charge_da = Constraint(model.T, rule=charge_constraint_da)

		def discharge_constraint_da(model, t):
			return model.energy_out_da[t] * self.hour_resolution <= (model.pwr)

		model.discharge_da = Constraint(model.T, rule=discharge_constraint_da)

		# because of the design of DRL, the first day no da market is available
		def energy_in_da_zero(model, t):
			if t < 96:
				return model.energy_in_da[t] == 0
			else:
				return Constraint.Skip

		def energy_out_da_zero(model, t):
			if t < 96:
				return model.energy_out_da[t] == 0
			else:
				return Constraint.Skip

		model.energy_in_da_zero = Constraint(model.T, rule=energy_in_da_zero)
		model.energy_out_da_zero = Constraint(model.T, rule=energy_out_da_zero)
		def energy_da_out_equal(model, t):
			if t % 4 != 3:
				return model.energy_out_da[t] == model.energy_out_da[t+1]
			else:
				return Constraint.Skip


		def energy_da_in_equal(model, t):
			if t % 4 != 3:
				return model.energy_in_da[t] == model.energy_in_da[t+1]
			else:
				return Constraint.Skip

		model.energy_da_out_equal = Constraint(model.T, rule=energy_da_out_equal)
		model.energy_da_in_equal = Constraint(model.T, rule=energy_da_in_equal)

		def timeseries_profit(model, t):
			current_profit = ((model.energy_out_id[t] * model.price_id[t] + model.energy_out_da[t] * model.price_da[t]) * discharge_efficiency - (model.energy_in_id[t] * model.price_id[t] + model.energy_in_da[t] * model.price_da[t]) / charging_efficiency)
			return model.profit_timeseries[t] == current_profit  

		model.profit_track = Constraint(model.T, rule=timeseries_profit)

		# use constraint to calculate cumulative cycle rate at each timestep
		def cycle_rate_per_ts(model, t):
			ts_cycle = ((model.energy_out[t] + model.energy_in[t]) / self.pwr) / 2
			if t == 0:
				return model.cumlative_cycle_rate[t] == ts_cycle + current_cycle_num
			else:
				return model.cumlative_cycle_rate[t] == model.cumlative_cycle_rate[t-1] + ts_cycle

		model.cycles = Constraint(model.T, rule=cycle_rate_per_ts)

		# define constraint for cumlative profit
		def cumlative_profit(model, t):
			if t == 0:
				return model.cumulative_profit[t] == (model.energy_out_id[t] * model.price_id[t] + model.energy_out_da[t] * model.price_da[t]) * discharge_efficiency  - ((model.energy_in_id[t] * model.price_id[t] + model.energy_in_da[t] * model.price_da[t]) / charging_efficiency)
			else:
				return model.cumulative_profit[t] == model.cumulative_profit[t-1] + ((model.energy_out_id[t] * model.price_id[t] + model.energy_out_da[t] * model.price_da[t]) * discharge_efficiency - ((model.energy_in_id[t] * model.price_id[t] + model.energy_in_da[t] * model.price_da[t]) / charging_efficiency))

		model.all_profit = Constraint(model.T, rule=cumlative_profit)

		# calulcate degradation costs
		if previous_ep_power != 0:
			alpha_degradation = ((((self.previous_cap - remaining_capacity)/100) * self.cap) / previous_ep_power) * self.batt_cost
		else:
			alpha_degradation = 0 

		# get degradation cost
		degrade_cost = [alpha_degradation * (model.energy_out[t] + model.energy_in[t]) for t in model.T]
		# define profit
		export_revenue = [price_df.iloc[t, -1] * model.energy_out[t] * discharge_efficiency for t in model.T]
		import_cost = [price_df.iloc[t, -1] * model.energy_in[t] / charging_efficiency for t in model.T]
		profit_ts = np.array(export_revenue) - np.array(import_cost) - np.array(degrade_cost)

		profit_obj = np.sum(profit_ts)

		degrade_cost_new = [alpha_degradation * (model.energy_out_id[t] + model.energy_out_da[t] + model.energy_in_id[t] + model.energy_in_da[t]) for t in model.T]
		export_revenue_new = [(prices_id[t] * model.energy_out_id[t] + prices_da[t] * model.energy_out_da[t]) * discharge_efficiency for t in model.T]
		import_cost_new = [(prices_id[t] * model.energy_in_id[t] + prices_da[t] * model.energy_in_da[t]) / charging_efficiency for t in model.T]
		profit_ts_new = np.array(export_revenue_new) - np.array(import_cost_new) - np.array(degrade_cost_new)

		profit_obj_new = np.sum(profit_ts_new)

		# new profit Johannes + Norman

		profit_obj_JJ_NZ = model.cumulative_profit[len(price_df)-1]


		# declare objective function
		# model.objective = Objective(expr=profit_obj, sense=maximize)
		model.objective = Objective(expr=profit_obj_JJ_NZ, sense=maximize)

		# implement bigM constraint to ensure model doesn't simultaneously charge and discharge
		def Bool_char_rule_1(model, t):
		    bigM=5000000
		    return((model.energy_in[t])>=-bigM*(model.charge_bool[t]))

		model.Batt_ch1=Constraint(model.T,rule=Bool_char_rule_1)

		# if battery is charging, charging must be greater than -large
		# if not, charging geq zero
		def Bool_char_rule_2(model, t):
		    bigM=5000000
		    return((model.energy_in[t])<=0+bigM*(1-model.discharge_bool[t]))

		model.Batt_ch2=Constraint(model.T,rule=Bool_char_rule_2)

		# if batt discharging, charging must be leq zero
		# if not, charging leq +large
		def Bool_char_rule_3(model, t):
		    bigM=5000000
		    return((model.energy_out[t])<=bigM*(model.discharge_bool[t]))

		model.Batt_cd3=Constraint(model.T,rule=Bool_char_rule_3)

		# if batt discharge, discharge leq POSITIVE large
		# if not, discharge leq 0
		def Bool_char_rule_4(model, t):
		    bigM=5000000
		    return((model.energy_out[t])>=0-bigM*(1-model.charge_bool[t]))

		model.Batt_cd4=Constraint(model.T,rule=Bool_char_rule_4)

		# if batt charge, discharge geq zero
		# if not, discharge geq -large
		def Batt_char_dis(model, t):
		    return (model.charge_bool[t]+model.discharge_bool[t]<=1)

		model.Batt_char_dis=Constraint(model.T,rule=Batt_char_dis)

		# declare molde solver and solve
		sol = SolverFactory('gurobi')
		sol.solve(model)

		return model

	def update_remaining_cap(self, cycle_num):
		remaining_cap = calculate_degradation(cycle_num)
		return remaining_cap


price_data = pd.read_csv('../../data/processed_data/DE_DA_QH_ID_test_milp_adjusted_010_090.csv')

# declare battery config
battery_power = 10 # vermutlich in MW
battery_capacity = 20 # vermutlich in MWh

# declare intial soc
soc = 0.5 * battery_capacity
current_cycle = 0
remaining_capacity = 100
previous_ep_power = 0
#horizon = 168
horizon = 292 * 24 * 4

# Instaniate MILP battery object with price data
battery = BatteryMILP(battery_power, battery_capacity)

start_time = time.time()
# pass daily prices for optmisation
battery_solved = battery.optimise(price_data, soc, current_cycle, remaining_capacity, previous_ep_power)

model_results = {}

for idx, v in enumerate(battery_solved.component_objects(Var, active=True)):
	# print(idx, v.getname())

	var_val = getattr(battery_solved, str(v))

	model_results[f'{v.getname()}'] = var_val[:].value

df = pd.DataFrame(model_results)

end_time = time.time()
print(f'Runtime: {end_time - start_time}')

# update cumulative profit so ensure profits carried between episodes
df['cumulative_profit_check'] = df['profit_timeseries'].cumsum()
df["price_id"] = price_data["price_id"].values
df["price_da_qh"] = price_data["price_da_qh"].values

# save profits for runtime duration (for comparison with DQN models)
df.to_csv('../results/timeseries_results_MILP_da_id_pyomo.csv')

plt.plot(df['cumulative_profit_check'].values)
plt.show()

df.to_clipboard()