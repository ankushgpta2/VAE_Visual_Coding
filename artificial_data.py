import numpy as np
import pandas as pd
import random as rd
import networkx as nx
import matplotlib.pyplot as plt
import math
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from vae_structure import *
import networkx as nx
import collections
from run_artificial_through_dnn import *

def artificial_main(repeat):
    artificial_parameters = get_artificial_params()
    list_of_cells = generate_baseline_activity(artificial_parameters)
    activity_ramp, frames_per_AP_event = generate_response_profile(artificial_parameters, list_of_cells)
    cell_to_grating_tuning, frames_per_orientation, random_gratings, on_set, list_of_gratings, stim_duration_in_frames, frames_per_trial = get_neurons_and_frames(artificial_parameters)
    list_of_cells = introduce_noise_to_baseline(artificial_parameters, list_of_cells)
    global_activation_order, ensemble_holder, positions = get_initial_active_and_branch(artificial_parameters, cell_to_grating_tuning)
    list_of_cells, frames_per_orientation = modify_baseline_with_stim_response(global_activation_order, random_gratings, frames_per_orientation, on_set, list_of_cells,
                                                       artificial_parameters, frames_per_AP_event, activity_ramp)
    general_parameters, mlp_parameters, vae_parameters, lstm_parameters, validation_data, validation_labels, \
                test_labels, test_data, input_labels, input_data, artificial_df, data_per_trial = prep_data_for_dnn(list_of_cells, artificial_parameters, list_of_gratings, frames_per_orientation,
                                                                                                                    on_set, frames_per_trial, random_gratings)
    run_actual_neural_nets(general_parameters, mlp_parameters, vae_parameters, artificial_parameters, validation_data, validation_labels, test_labels, test_data, input_labels, input_data,
                               data_per_trial, random_gratings)


def get_artificial_params():
    # parameters
    # might want to assume sparse activity with somewhat homogenous spatial spread of neurons that are tuned for each grating + higher level gamma synchronization
    num_cells = 500
    num_time_bins = 10000  # number of frames essentially... assumes acquisition resolution (no temporal coarse graining)
    stim_duration = 1  # in seconds
    gap_until_next = 7  # in seconds
    different_classes = 4
    tuned_cell_range = [0.05, 0.3]
    delay_after_stim = 0.1  # in seconds
    imaging_rate = 45 # in hertz
    synaptic_delay = 0.1  # in seconds, this would be a 5 frame delay
    data_split = [0.6, 0.4]  # how to split the simulated dataset
    single_gen_branching = True  # whether the first cell in the list of tuned cells splits for 2nd and 3rd and then there is no more branching
    random_branching = False  # splits activity, reflected in synaptic delay, randomly
    num_of_initial_cell_activations = [1/num_cells, 0.2]  # proportion of tuned cells
    branching_after_how_many_gen = 'hold'  # this is a future parameter for how often to branch... how many generations leads to split
    ensemble_level_temporal_profile = 'hold'  # this specifies the higher level temporal relationship between multiple cells that are coactivated during stimulus presentation
    spike_rate_of_neurons = 10  # how many APs are triggered in response to stimulus, APs/stimulus duration
    vary_spike_rate = 'hold'
    random_event_prob = 0.01  # how often random spike events will be introduced... proportion of (total frames or time bins for random / spike rate of neuron) / total frames
    spatial_spread_of_random_events = [0.1, 0.5]  # range of how many neurons will be influenced by the random event
    artificial_parameters = {'num_cells': num_cells, 'num_time_bins': num_time_bins, 'stim_duration': stim_duration, 'gap_until_next': gap_until_next, 'different_classes': different_classes,
                         'tuned_cell_range': tuned_cell_range, 'delay_after_stim': delay_after_stim, 'imaging_rate': imaging_rate, 'synaptic_delay': synaptic_delay, 'data_split': data_split,
                         'single_gen_branching': single_gen_branching, 'random_branching': random_branching, 'num_of_initial_cell_activations': num_of_initial_cell_activations,
                         'branching_after_how_many_gen': branching_after_how_many_gen, 'ensemble_level_temporal_profile': ensemble_level_temporal_profile, 'spike_rate_of_neurons': spike_rate_of_neurons,
                         'vary_spike_rate': vary_spike_rate, 'random_event_prob': random_event_prob, 'spatial_spread_of_random_events': spatial_spread_of_random_events}
    return artificial_parameters


def generate_baseline_activity(artificial_parameters):
    # -------------------------------------------------- SPECIFY THE BASELINE ACTIVITY FOR EACH NEURON IN NETWORK ---------------------------------------------------
    num_cells = artificial_parameters['num_cells']
    num_time_bins = artificial_parameters['num_time_bins']
    # first establish baseline activity --> interaction between cells across time = gaussian + individual ROI change across time = gaussian
    list_of_cells = np.empty((num_time_bins, num_cells))
    for x in range(num_time_bins):
        list_of_cells[x, :num_cells] = np.random.normal(size=(1, num_cells))  # gaussian distribution as baseline activity in network

    # normalize the cells activity through frames --> between 0 and 1
    abs_minimum = abs(list_of_cells.min())
    for x in range(num_cells):
        list_of_cells[:, x] = list_of_cells[:, x] + abs_minimum
    for x in range(num_cells):
        list_of_cells[:, x] = list_of_cells[:, x] / list_of_cells.max()
    return list_of_cells


def generate_response_profile(artificial_parameters, list_of_cells):
    # -------------------------------------------------- ELICITED RESPONSE PROFILE FROM STIMULUS ---------------------------------------------------
    stim_duration = artificial_parameters['stim_duration']
    spike_rate_of_neurons = artificial_parameters['spike_rate_of_neurons']
    imaging_rate = artificial_parameters['imaging_rate']
    num_time_bins = artificial_parameters['num_time_bins']
    delay_after_stim = artificial_parameters['delay_after_stim']
    # specify the decrease in activity --> each frame is 22ms
    lower_bound = 0.4
    upper_bound = 1.0
    activity_decrease_step = 0.1  # this refers to the spiking rate essentially or general level of activity... it is spiking faster than acquisition rate
    frames_per_AP_event = int((stim_duration / spike_rate_of_neurons) * imaging_rate)

    num_of_steps = round((upper_bound - lower_bound) / 0.1)
    activity_vals = []
    counter = upper_bound
    for x in range(num_of_steps-1):
        if counter > lower_bound:
            if counter == upper_bound:
                activity_value = np.random.uniform(counter-0.1, counter+0.1)
                counter -= activity_decrease_step + 0.3
            else:
                activity_value = np.random.uniform(counter-0.15, counter+0.15)
                counter -= activity_decrease_step
            activity_vals.append(activity_value)

    frame_counter = 100
    activity_ramp = []
    for x in activity_vals:
        subset = example[frame_counter:frame_counter + frames_per_AP_event]
        local_average = np.mean(example[frame_counter:frame_counter+frames_per_AP_event])
        final_activity_value = x / local_average
        activity_ramp.append(final_activity_value)
        example[frame_counter:frame_counter + frames_per_AP_event] = subset * final_activity_value
        example[frame_counter+400:frame_counter + 400 + frames_per_AP_event] = example[frame_counter+400:frame_counter + 400 + frames_per_AP_event] * final_activity_value
        example[frame_counter+800:frame_counter + 800 + frames_per_AP_event] = example[frame_counter+800:frame_counter + 800 + frames_per_AP_event] * final_activity_value
        if x == activity_vals[0]:
            axes[1].plot([frame_counter, frame_counter + round(imaging_rate*0.1)], [0.4, 0.4], color='black', label='First 100ms', linewidth=5)
            axes[1].plot([frame_counter+400, frame_counter + round(imaging_rate*0.1) + 400], [0.4, 0.4], color='black', linewidth=5)
            axes[1].plot([frame_counter + 800, frame_counter + round(imaging_rate * 0.1) + 800], [0.4, 0.4], color='black', linewidth=5)
        frame_counter += frames_per_AP_event

    return activity_ramp, frames_per_AP_event


def get_neurons_and_frames(artificial_parameters):
    num_cells, different_classes, tuned_cell_range, stim_duration, imaging_rate, gap_until_next, num_time_bins = artificial_parameters['num_cells'], artificial_parameters['different_classes'], \
                artificial_parameters['tuned_cell_range'], artificial_parameters['stim_duration'], artificial_parameters['imaging_rate'], artificial_parameters['gap_until_next'], \
                artificial_parameters['num_time_bins']
    # -------------------------------------------------- ORIENTATION GRATINGS INFO: CELLS --> CLASS + FRAMES --> CLASS ---------------------------------------------------
    # pair certain cells within the network to be tuned for certain orientation grating
    keylist = [x for x in range(different_classes)]
    cell_to_grating_tuning = {}
    for i in keylist:
        cell_to_grating_tuning[i] = []

    for i in range(different_classes):
        random_num_of_cells = np.random.randint(round(num_cells * tuned_cell_range[0]), round(num_cells * tuned_cell_range[1]))
        cell_list = []
        for x in range(random_num_of_cells):
            selected_cell_for_tuning = np.random.randint(0, num_cells)
            cell_list.append(selected_cell_for_tuning)
        cell_to_grating_tuning[i] = cell_list

    # determine frames for actual stimulus presentation
    stim_duration_in_frames = int(np.floor(stim_duration * imaging_rate))
    off_periods_in_frames = round(imaging_rate * gap_until_next)
    total_frames_per_trial = stim_duration_in_frames + off_periods_in_frames

    on_set = []
    off_set = []
    i = 0
    while i <= num_time_bins:
        ending_of_stimulus = stim_duration_in_frames + i
        on_set.append([x for x in range(i, ending_of_stimulus)])
        off_set.append([x for x in range(ending_of_stimulus, ending_of_stimulus + off_periods_in_frames)])
        i += total_frames_per_trial

    # get orientation presentation per trial randomly
    num_trials = np.shape(on_set)[0]
    list_of_gratings = [x for x in range(different_classes)]
    if num_trials > different_classes:
        random_gratings = rd.choices(list_of_gratings, k=num_trials)
    else:
        random_gratings = rd.sample(list_of_gratings, num_trials)

    # initialize the dictionary that will store the frames that correspond to each type of orientation presentation
    frames_per_orientation = {}
    for i in list_of_gratings:
        frames_per_orientation[i] = None     # map the frames to each grating for dataframe
    return cell_to_grating_tuning, frames_per_orientation, random_gratings, on_set, list_of_gratings, stim_duration_in_frames, total_frames_per_trial


def introduce_noise_to_baseline(artificial_parameters, list_of_cells):
    spatial_spread_of_random_events = artificial_parameters['spatial_spread_of_random_events']
    num_cells = artificial_parameters['num_cells']
    # ---------------------------------------------- BASED ON SELECTED CELLS AND FRAMES... MODIFY BASELINE WITH ACTIVITY PROFILE ----------------------------------------------
    # introduce spontaneous spiking activity or oscillations into this baseline for random cells in the network
    # assumes that the random event happens at the same time for all of the neurons and that it is either an increase or decrease in activity for between 50-200 frames
    range_of_rand_event = [10, 50]  # in frames... with 50Hz, this is between 0.2 and 1 seconds
    random_activity_factor = [1.05, 1.5]  # 1.5 times baseline activity for event duration
    timing_of_events = np.random.poisson(lam=artificial_parameters['random_event_prob'], size=artificial_parameters['num_time_bins'])  # gets the location of a random event throughout recording
    frame_counter = 0
    for x in timing_of_events:
        if x == 1:
            duration_of_rand_event = np.random.randint(range_of_rand_event[0], range_of_rand_event[1])
            random_subset = rd.randrange(round(spatial_spread_of_random_events[0]*num_cells), round(spatial_spread_of_random_events[1]*num_cells))
            activity_factor = rd.uniform(random_activity_factor[0], random_activity_factor[1])
            selected_cells_for_random = []
            for y in range(random_subset):
                selected_cells_for_random.append(np.random.choice(range(0, num_cells), replace=False))
            for i in range(len(selected_cells_for_random)):
                # determine whether to lower or increase the activity by the random_activity_factor
                random_operator = rd.randint(0, 1)
                activity_to_be_mod = list_of_cells[frame_counter:frame_counter + duration_of_rand_event, selected_cells_for_random[i]]
                if random_operator == 0:
                    mod_activity = activity_to_be_mod * activity_factor
                elif random_operator == 1:
                    mod_activity = activity_to_be_mod / activity_factor
                list_of_cells[frame_counter:frame_counter + duration_of_rand_event, selected_cells_for_random[i]] = mod_activity
        frame_counter += 1
        return list_of_cells


def get_initial_active_and_branch(artificial_parameters, cell_to_grating_tuning):
    single_gen_branching, random_branching, num_of_initial_cell_activations, branching_after_how_many_gen = artificial_parameters['single_gen_branching'], artificial_parameters['random_branching'], \
                                                                                                            artificial_parameters['num_of_initial_cell_activations'], \
                                                                                                           artificial_parameters['branching_after_how_many_gen']
    num_of_initial_cell_activations = artificial_parameters['num_of_initial_cell_activations']
    global_activation_order = {}
    for x in range(artificial_parameters['different_classes']):
        cells_for_grating = cell_to_grating_tuning[x]
        initial_activation_num = rd.uniform(num_of_initial_cell_activations[0], num_of_initial_cell_activations[1])
        num_initial = initial_activation_num * len(cells_for_grating)
        if num_initial < 1:
            num_initial = 1
        else:
            num_initial = round(num_initial)
        activation_order_for_cells = {0: cells_for_grating[:num_initial]}
        if single_gen_branching is True:
            first_branch_units = cells_for_grating[num_initial:num_initial+(num_initial*2)]
            activation_order_for_cells[1] = first_branch_units
            starting_for_rem = len(first_branch_units)+num_initial
            rem_follower_units = cells_for_grating[starting_for_rem:starting_for_rem + (len(cells_for_grating) - starting_for_rem)]
            forward_equal_sequences = np.floor(len(rem_follower_units) / len(first_branch_units))
            last_rem_units = len(cells_for_grating) - (num_initial + len(first_branch_units) + forward_equal_sequences * len(first_branch_units))
            counter = 0
            for i in range(2, int(forward_equal_sequences)+2):
                next_activations = rem_follower_units[counter:counter+len(first_branch_units)]
                activation_order_for_cells[i] = next_activations
                counter += len(first_branch_units)
            if last_rem_units > 0:
                activation_order_for_cells[int(forward_equal_sequences + 2)] = rem_follower_units[int(forward_equal_sequences)*len(first_branch_units):]
        else:
            key_counter = 1
            for z in range(num_initial, len(cells_for_grating), num_initial):
                activation_order_for_cells[key_counter] = cells_for_grating[z:z+num_initial]
                key_counter += 1
        global_activation_order[x] = activation_order_for_cells
    ensemble_holder, positions = spatially_group_cells(global_activation_order)
    return global_activation_order, ensemble_holder, positions

def modify_baseline_with_stim_response(global_activation_order, random_gratings, frames_per_orientation, on_set, list_of_cells, artificial_parameters, frames_per_AP_event, activity_ramp):
    delay_after_stim, imaging_rate, synaptic_delay, spike_rate_of_neurons = artificial_parameters['delay_after_stim'], artificial_parameters['imaging_rate'], artificial_parameters['synaptic_delay'], \
                                                                                                artificial_parameters['spike_rate_of_neurons']
    # modify neuron activities based on tuning preference with the ramped activity profile
    for x in range(len(random_gratings)):
        cells_for_grating = global_activation_order[random_gratings[x]]
        # print(str(len(cells_for_grating)) + '/' + str(num_cells) + ' Cells Tuned Towards ' + str(random_gratings[x]))
        frames_for_trial = on_set[x][:]
        # getting the frames for each trial and matching to the orientation gratings... the else handles appending frames for multiple trials for same class
        if frames_per_orientation[random_gratings[x]] is None:
            frames_per_orientation[random_gratings[x]] = frames_for_trial
        else:
            appended_frames = frames_per_orientation[random_gratings[x]] + frames_for_trial
            frames_per_orientation[random_gratings[x]] = appended_frames
        synaptic_delay_per_neuron = 0
        # getting number of subplots for one of the trials to plot
        if x == 20:
            if len(cells_for_grating.keys()) > 10:
                num_subplots = 10
            else:
                num_subplots = len(cells_for_grating.keys())
            fig, axes = plt.subplots(num_subplots, 1)
            plt.suptitle('Simulated Neural Activity For Follower Units In Network', fontweight='bold')
            starting_frame = frames_for_trial[0] - 200
            ending_frame = frames_for_trial[0] + 200
        # cycle through each of the activation sets for the particular orientation grating
        for i in range(len(cells_for_grating.keys())):
            cells_for_specific_activation = cells_for_grating[i]
            # cycle through the cells within a specific activation set
            for y in cells_for_specific_activation:
                if i == 0:
                    synaptic_delay_per_neuron = 0
                    start_frame = frames_for_trial[0] + round((delay_after_stim * imaging_rate))  # the first neuron is delayed by only stimulus onset lag
                else:
                    synaptic_delay_per_neuron += round(synaptic_delay * imaging_rate)
                    start_frame = frames_for_trial[0] + round((delay_after_stim * imaging_rate)) + synaptic_delay_per_neuron
                # for each cell apply the multiplier per number of frames for each "AP Event"
                frame_counter = start_frame
                for z in activity_ramp:
                    subset_for_cell = list_of_cells[start_frame:start_frame+frames_per_AP_event, y]
                    multiplied = subset_for_cell * z
                    list_of_cells[start_frame:start_frame+frames_per_AP_event, y] = multiplied
                    frame_counter += frames_per_AP_event
            # plot one cell from each activation cohort for the trial (20th trial)
                if x == 20 and i < 10:
                    axes[i].plot(range(starting_frame, ending_frame), list_of_cells[starting_frame:ending_frame, cells_for_specific_activation[0]], color='dodgerblue')
                    starting_frame_for_activation_set = frame_counter-(frames_per_AP_event * len(activity_ramp))+synaptic_delay_per_neuron
                    axes[i].plot(range(start_frame-(round(synaptic_delay*imaging_rate))-(round(delay_after_stim*imaging_rate)), frame_counter),
                                 list_of_cells[start_frame-(round(synaptic_delay*imaging_rate))-(round(delay_after_stim*imaging_rate)):frame_counter, cells_for_specific_activation[0]], color='red')
                    axes[i].plot([frames_for_trial[0], frames_for_trial[0]], [0, 1.7], '--', color='black', linewidth=2)
                    axes[i].set_ylim([0, 1.5])
                    if i != num_subplots-1:
                        axes[i].set_xticks([])
        # plt.show()
    return list_of_cells, frames_per_orientation
