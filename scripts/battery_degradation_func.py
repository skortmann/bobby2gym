import numpy as np
import matplotlib.pyplot as plt


def calculate_degradation(cycle_num):
    ''' representative degradation curve derived from https://www.researchgate.net/publication/303890624_Modeling_of_Lithium-Ion_Battery_Degradation_for_Cell_Life_Assessment'''
    remaining_cap = 0.000000000000000000093031387249 * cycle_num ** 6 - 0.00000000000000135401195479516 * cycle_num ** 5 + 0.00000000000769793660702074 * cycle_num ** 4 - 0.0000000215754356367543 * cycle_num ** 3 + 0.000031144155854923 * cycle_num ** 2 - 0.025873264406556 * cycle_num + 100
    return remaining_cap


if __name__ == '__main__':
    # Generate cycle numbers from 0 to 1000
    cycle_num = np.linspace(0, 4000, 1000)
    # Calculate degradation for each cycle number
    degradation_curve = calculate_degradation(cycle_num)
    # store the data with czcle number as x axis and degradation curve as y axis as a csv file in results csv under battery_degradation.csv
    # can u save only every 20th data point in the csv file
    np.savetxt('../results/CSV/battery_degradation.csv', np.column_stack((cycle_num[::10], degradation_curve[::10])), delimiter=',', header='cycle_num, remain_cap', comments='', fmt='%s')

    # Plotting the degradation curve
    plt.figure(figsize=(10, 6))
    plt.plot(cycle_num, degradation_curve, label="Battery Capacity Degradation")
    plt.xlabel("Cycle Number")
    plt.ylabel("Remaining Capacity (%)")
    plt.title("Lithium-Ion Battery Capacity Degradation Over 1000 Cycles")
    plt.legend(loc="upper right", facecolor='lightgray')
    plt.grid(True)
    plt.show()
