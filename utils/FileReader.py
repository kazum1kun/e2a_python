import json
import re

import numpy as np


# Read activities from a file
# The file has the following format
# int
# int int
# The first line contains an integer N, the next N lines each contain integer pairs (time index)
# where index is between 1 and 21 (the index of the user activity)
# and time is the time this activity happened.
#
# Output: an N+1 by 2 array where the first row is all zeros (to maintain 1-based indexing)
#                                and following rows has (time index) format as in the input
def read_activities(file):
    with open(file, 'r') as act_file:
        # Read the first line to obtain the number of lines
        num_activities = int(act_file.readline())
        activities = np.zeros((num_activities + 1, 2), np.int_)
        i = 1

        for line in act_file:
            activities[i][0] = int(line.split(' ')[0])
            activities[i][1] = int(line.split(' ')[1])
            i += 1

        return activities


# Read device events from a file
# The file has the following format
# int
# int char
# The first line contains an integer M, the next M lines each contain a pair (time, event)
# where event is a character between 'A' and 'Z',
# and time is the time (in seconds) that this event happened.
#
# Output: a M+1 by 2 array where the first row is all zeros (to maintain 1-based indexing)
#                                and following rows has (time event) format as in the input
# NOTE: the given file has time in float, which is converted to int upon reading
# NOTE 2: event letters are converted to their ASCII values to facilitate processing
def read_events(file, dtype='int'):
    with open(file, 'r') as event_file:
        # Read the first line to obtain the number of lines
        num_events = int(event_file.readline())
        if dtype == 'int':
            events = np.zeros((num_events + 1, 2), np.int_)
        elif dtype == 'float':
            events = np.zeros((num_events + 1, 2), np.float_)
        i = 1

        for line in event_file:
            tokens = line.strip().split(' ')
            events[i][0] = float(tokens[0])
            events[i][1] = ord(tokens[1])
            i += 1

        return events


# Read activity-to-events mapping from a file
# The file has the following format
# int
# int char...
# Where the first line contains an integer X, the next X lines contain a list (activity, event, ...)
# There can be as little as one event corresponding to an activity, and there is no upper limit on # of events
# there can be multiple identical activity entries, meaning they can trigger a subset of events (Sk) in stead of
# the full sequence (S1). The full sequence is always listed first
#
# Output: a dictionary where keys corresponds to activities, and values are lists of lists, each sublist containing
# a sequence of events triggered by this activity. l[0] are empty and l[1] is always the full sequence
# NOTE: the event letters are converted to ASCII values to facilitate processing
def read_mappings(file):
    with open(file, 'r') as map_file:
        # Skip the first line
        map_file.readline()
        mappings = {}

        for line in map_file:
            # The first entry is an activity and the following entries are the resulting events
            act, sub, *events = line.strip().split(' ')
            act = re.findall(r'\d+', act)
            act = int(act[0])
            events = [ord(e) for e in events]
            events.insert(0, -1)

            if act not in mappings:
                mappings[act] = [[], events]
            else:
                mappings[act].append(events)

        return mappings


# Same as read_mappings but without the added pseudo-entries for 1-based arrays
def read_mappings_0based(file):
    with open(file, 'r') as map_file:
        # Skip the first line
        map_file.readline()
        mappings = {}

        for line in map_file:
            # The first entry is an activity and the following entries are the resulting events
            act, sub, *events = line.strip().split(' ')
            act = re.findall(r'\d+', act)
            act = int(act[0])

            mappings[0] = []

            if act not in mappings:
                mappings[act] = [events]
            else:
                mappings[act].append(events)

        return mappings


# Read the device-event mappings
def read_device_event(file):
    with open(file, 'r') as map_file:
        mappings = {}
        for line in map_file:
            act, *events = line.strip().split(' ')

            if act not in mappings:
                mappings[act] = events
            else:
                mappings[act].append(events)

    return mappings


# Read in events as list of lists
def read_mappings_list(file):
    with open(file, 'r') as map_file:
        map_file.readline()
        mappings = []
        for line in map_file:
            idx, sub_idx, *activities = line.strip().split(' ')
            mappings.append([(int(re.findall(r'\d+', idx)[0]), int(re.findall(r'\d+', sub_idx)[0])), tuple(activities)])
        return mappings


# Read in a json file and return it as a dict
def read_json(file):
    with open(file) as json_file:
        return json.load(json_file)
