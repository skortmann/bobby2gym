from gurobipy import *
import pandas as pd
import numpy as np

def solve(df, soc_initial, battery_power, soc_minimal=0, soc_maximal=1,first_day_no_da=False, daywise_da=False, soc_final=0.5, energy_in_da=None, energy_out_da=None, energy_in_id=None, energy_out_id=None):
    model = Model('MILP')

    T = len(df)
    price_id = df.price_id.values
    price_da = df.price_da_qh.values

    energy = {}
    soc = {}
    x = {}
    y = {}

    for i in range(T):
        for j in ["in", "out"]:
            for k in ["sum", "da", "id"]:
                energy[i, j, k] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"energy_{i}_{j}_{k}")
            x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"charge_{i}_{j}")
            y[i, j] = model.addVar(vtype=GRB.BINARY, name=f"charge_da_{i}_{j}")
            for k in ["sum", "da"]:
                model.addConstr(energy[i, j, k] * 4 <= battery_power)
        soc[i] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"soc_{i}")
        model.addConstr(soc[i] <= soc_maximal, name=f"soc_max_{i}")
        model.addConstr(soc[i] >= soc_minimal, name=f"soc_min_{i}")
        model.addConstr(energy[i, "out", "sum"] >= energy[i, "out", "da"] + energy[i, "out", "id"] - energy[i, "in", "da"] - energy[i, "in", "id"], name=f"out_sum_{i}")
        model.addConstr(energy[i, "in", "sum"] >= energy[i, "in", "da"] + energy[i, "in", "id"] - energy[i, "out", "da"] - energy[i, "out", "id"], name=f"in_sum_{i}")
        model.addConstr(energy[i, "out", "sum"] <= energy[i, "out", "da"] + energy[i, "out", "id"], name=f"energy_in_sum_{i}")
        model.addConstr(energy[i, "in", "sum"] <= energy[i, "in", "da"] + energy[i, "in", "id"], name=f"energy_out_sum_{i}")

        if i == 0:
            model.addConstr(soc[i] == soc_initial + energy[i, "in", "da"] + energy[i, "in", "id"] - energy[i, "out", "da"] - energy[i, "out", "id"], name=f"soc_{i}")
        else:
            model.addConstr(soc[i] == soc[i-1] + energy[i, "in", "da"] + energy[i, "in", "id"] - energy[i, "out", "da"] - energy[i, "out", "id"], name=f"soc_{i}")
        # added to iterative model
        if i == T - 1:
            model.addConstr(soc[i] == soc_final, name=f"soc_final")

        model.addConstr(x[i, "in"] + x[i, "out"] <= 1, name=f"charge_{i}")
        model.addConstr(energy[i, "in", "sum"] * 4 <= x[i, "in"] * battery_power, name=f"charge_in_{i}")
        model.addConstr(energy[i, "out", "sum"] * 4 <= x[i, "out"] * battery_power, name=f"charge_out_{i}")

        model.addConstr(y[i, "in"] + y[i, "out"] <= 1, name=f"charge_da_{i}")
        model.addConstr(energy[i, "in", "da"] * 4 <= y[i, "in"] * battery_power, name=f"charge_da_in_{i}")
        model.addConstr(energy[i, "out", "da"] * 4 <= y[i, "out"] * battery_power, name=f"charge_da_out_{i}")

        model.addConstr(energy[i, "in", "id"] * 4 <= 2 * battery_power, name=f"max_charge_id_in_{i}")
        model.addConstr(energy[i, "out", "id"] * 4 <= 2 * battery_power, name=f"max_charge_id_out_{i}")
    # add the constraint making day ahead trading available the second day.
    if first_day_no_da:
        for i in range(24*4):
            for j in ["in", "out"]:
                model.addConstr(energy[i, j, "da"] == 0, name=f"first_day_no_day_ahead_{i}_{j}")
    # Assuming you have a dictionary of variables 'x' and a model 'm'
    for i in range(0, T, 4):
        for j in ["in", "out"]: # T is the total number of time periods
            model.addConstr(energy[i, j, "da"] == energy[i + 1, j, "da"], "c1_" + str(i))
            model.addConstr(energy[i + 1, j, "da"] == energy[i + 2, j, "da"], "c2_" + str(i))
            model.addConstr(energy[i + 2, j, "da"] == energy[i + 3, j, "da"], "c3_" + str(i))

    # want to set
    if daywise_da:
        for i in range(96, T, 24*4):
            model.addConstr(soc[i-1] == soc_initial, name=f"daywise_da_{i}")
    if energy_out_da is not None:
        for i in range(T):
            model.addConstr(energy[i, "out", "da"] == energy_out_da[i], name=f"da_schedule__out_{i}")
    if energy_in_da is not None:
        for i in range(T):
            model.addConstr(energy[i, "in", "da"] == energy_in_da[i], name=f"da_schedule_in_{i}")
    if energy_out_id is not None:
        for i in range(T):
            model.addConstr(energy[i, "out", "id"] == energy_out_id[i], name=f"id_schedule_out_{i}")
    if energy_in_id is not None:
        for i in range(T):
            model.addConstr(energy[i, "in", "id"] == energy_in_id[i], name=f"id_schedule_in_{i}")

    # objective
    objective = quicksum(price_da[i] * (energy[i, "out", "da"] - energy[i, "in", "da"]) + price_id[i] * (energy[i, "out", "id"] - energy[i, "in", "id"]) for i in range(T))


    model.setObjective(objective, GRB.MAXIMIZE)
    # model.write("Norman_MILP.lp")

    model.optimize()

    # Initialize an empty DataFrame with 't' as the index

    # After model.optimize()
    # Initialize an empty DataFrame with 't' as the index
    variables_df = pd.DataFrame(index=range(T))

    # After model.optimize()
    if model.status == GRB.OPTIMAL:
        for v in model.getVars():
            if "energy" in v.varName:
                i, j, k = v.varName.split("_")[1:]
                variables_df.loc[int(i), f"energy_{j}_{k}"] = v.x
            elif "charge" in v.varName:
                if 'da' not in v.varName:
                    i, j = v.varName.split("_")[1:]
                    variables_df.loc[int(i), f"charge_{j}"] = v.x
                else:
                    i, j = v.varName.split("_")[2:]
                    variables_df.loc[int(i), f"charge_da_{j}"] = v.x
            elif "soc" in v.varName:
                i = v.varName.split("_")[1]
                variables_df.loc[int(i), "soc"] = v.x
        variables_df["price_id"] = price_id
        variables_df["price_da"] = price_da

    # Return a DataFrame with the variables
    return variables_df






