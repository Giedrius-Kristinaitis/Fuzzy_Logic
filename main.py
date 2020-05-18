import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

# input variables
input_participants = 150
input_available_slots = 15
input_test_difficulty = 6

# create universes
participants = np.arange(50, 201, 1)
available_slots = np.arange(0, 21, 1)
test_difficulty = np.arange(0, 11, 1)
acceptance_probability = np.arange(0, 101, 1)

# create membership functions
participants_low = fuzz.trapmf(participants, [50, 50, 75, 110])
participants_medium = fuzz.trimf(participants, [90, 120, 150])
participants_high = fuzz.trapmf(participants, [130, 160, 200, 200])

available_slots_low = fuzz.trapmf(available_slots, [0, 0, 2, 8])
available_slots_medium = fuzz.trapmf(available_slots, [4, 8, 12, 16])
available_slots_high = fuzz.trapmf(available_slots, [13, 17, 20, 20])

test_difficulty_low = fuzz.trapmf(test_difficulty, [0, 0, 3, 5])
test_difficulty_medium = fuzz.trimf(test_difficulty, [4, 6, 8])
test_difficulty_high = fuzz.trapmf(test_difficulty, [7, 9, 10, 10])

acceptance_probability_low = fuzz.trimf(acceptance_probability, [0, 0, 50])
acceptance_probability_medium = fuzz.trimf(acceptance_probability, [10, 50, 90])
acceptance_probability_high = fuzz.trimf(acceptance_probability, [50, 100, 100])

# plot all universes
fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=4, figsize=(8, 9))

ax0.plot(participants, participants_low, 'b', linewidth=1.5, label='Mazas')
ax0.plot(participants, participants_medium, 'g', linewidth=1.5, label='Vidutinis')
ax0.plot(participants, participants_high, 'r', linewidth=1.5, label='Didelis')
ax0.set_title('Dalyviu skaicius')
ax0.legend()

ax1.plot(available_slots, available_slots_low, 'b', linewidth=1.5, label='Mazas')
ax1.plot(available_slots, available_slots_medium, 'g', linewidth=1.5, label='Vidutinis')
ax1.plot(available_slots, available_slots_high, 'r', linewidth=1.5, label='Didelis')
ax1.set_title('Laisvu vietu skaicius')
ax1.legend()

ax2.plot(test_difficulty, test_difficulty_low, 'b', linewidth=1.5, label='Zemas')
ax2.plot(test_difficulty, test_difficulty_medium, 'g', linewidth=1.5, label='Vidutinis')
ax2.plot(test_difficulty, test_difficulty_high, 'r', linewidth=1.5, label='Aukstas')
ax2.set_title('Testo sudetingumo lygis')
ax2.legend()

ax3.plot(acceptance_probability, acceptance_probability_low, 'b', linewidth=1.5, label='Maza')
ax3.plot(acceptance_probability, acceptance_probability_medium, 'g', linewidth=1.5, label='Normali')
ax3.plot(acceptance_probability, acceptance_probability_high, 'r', linewidth=1.5, label='Didele')
ax3.set_title('Priemimo tikimybe')
ax3.legend()

for ax in (ax0, ax1, ax2, ax3):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

plt.tight_layout()

# calculate membership function values
participants_value_low = fuzz.interp_membership(participants, participants_low, input_participants)
participants_value_medium = fuzz.interp_membership(participants, participants_medium, input_participants)
participants_value_high = fuzz.interp_membership(participants, participants_high, input_participants)

available_slots_value_low = fuzz.interp_membership(available_slots, available_slots_low, input_available_slots)
available_slots_value_medium = fuzz.interp_membership(available_slots, available_slots_medium, input_available_slots)
available_slots_value_high = fuzz.interp_membership(available_slots, available_slots_high, input_available_slots)

test_difficulty_value_low = fuzz.interp_membership(test_difficulty, test_difficulty_low, input_test_difficulty)
test_difficulty_value_medium = fuzz.interp_membership(test_difficulty, test_difficulty_medium, input_test_difficulty)
test_difficulty_value_high = fuzz.interp_membership(test_difficulty, test_difficulty_high, input_test_difficulty)

# apply rules
value_rule_1 = np.fmax(np.fmax(participants_value_high, available_slots_value_low), test_difficulty_value_high)
value_rule_2 = np.fmin(np.fmax(participants_value_medium, test_difficulty_value_medium), 1 - available_slots_value_low)
value_rule_3 = np.fmax(test_difficulty_value_low, np.fmax(1 - participants_value_high, 1 - available_slots_value_low))

acceptance_probability_value_low = np.fmin(value_rule_1, acceptance_probability_low)
acceptance_probability_value_medium = np.fmin(value_rule_2, acceptance_probability_medium)
acceptance_probability_value_high = np.fmin(value_rule_3, acceptance_probability_high)

# plot rule values
acceptance_probability_zero = np.zeros_like(acceptance_probability)

fig, ax0 = plt.subplots(figsize=(8, 3))

ax0.fill_between(acceptance_probability, acceptance_probability_zero, acceptance_probability_value_low, facecolor='b',
                 alpha=0.5)
ax0.plot(acceptance_probability, acceptance_probability_low, 'b', linewidth=0.5, linestyle='--', )
ax0.fill_between(acceptance_probability, acceptance_probability_zero, acceptance_probability_value_medium,
                 facecolor='g', alpha=0.5)
ax0.plot(acceptance_probability, acceptance_probability_medium, 'g', linewidth=0.5, linestyle='--')
ax0.fill_between(acceptance_probability, acceptance_probability_zero, acceptance_probability_value_high, facecolor='r',
                 alpha=0.5)
ax0.plot(acceptance_probability, acceptance_probability_high, 'r', linewidth=0.5, linestyle='--')
ax0.set_title('Priklausomybes funkciju rezultatai')

for ax in (ax0,):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

plt.tight_layout()

# aggregate probability values
aggregated = np.fmax(acceptance_probability_value_low,
                     np.fmax(acceptance_probability_value_medium, acceptance_probability_value_high))
plt.show()
# defuzzify result
probability = fuzz.defuzz(acceptance_probability, aggregated, 'centroid')
probability_activation = fuzz.interp_membership(acceptance_probability, aggregated, probability)  # for plot

# plot results
fig, ax0 = plt.subplots(figsize=(8, 3))

ax0.plot(acceptance_probability, acceptance_probability_low, 'b', linewidth=0.5, linestyle='--', )
ax0.plot(acceptance_probability, acceptance_probability_medium, 'g', linewidth=0.5, linestyle='--')
ax0.plot(acceptance_probability, acceptance_probability_high, 'r', linewidth=0.5, linestyle='--')
ax0.fill_between(acceptance_probability, acceptance_probability_zero, aggregated, facecolor='Orange', alpha=0.6)
ax0.plot([probability, probability], [0, probability_activation], 'k', linewidth=1.5, alpha=0.8)
ax0.set_title('Taisykliu ir defuzifikacijos rezultatai')

for ax in (ax0,):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

plt.tight_layout()
plt.show()

# print acceptance probability
print(probability)
