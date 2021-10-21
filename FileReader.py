import numpy as np


# Read activities from a file
# The file has the following format
# int
# int int
# int int
# The first line contains an integer N, the next N lines each contain integer pairs (time index)
# where index is between 1 and 21 (the index of the user activity)
# and time is the time this activity happened.
#
# Output: a N+1 by 2 array where the first row is all zeros (to maintain 1-based indexing)
#                                and following rows has (time index) format as in the input
def read_activities(file):
    with open(file, 'r') as act_file:
        # Read the first line to obtain the number of lines
        num_activities = int(act_file.readline())
        activities = np.zeros((num_activities + 1, 2), dtype=np.int_)
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
# int char
# The first line contains an integer M, the next M lines each contain a pair (time, event)
# where event is a character between 'A' and 'Z',
# and time is the time (in seconds) that this event happened.
#
# Output: a M+1 by 2 array where the first row is all zeros (to maintain 1-based indexing)
#                                and following rows has (time event) format as in the input
# NOTE: the given file has time in float, which is converted to int upon reading
# NOTE 2: event letters are converted to their ASCII values to facilitate processing
def read_events(file):
    with open(file, 'r') as event_file:
        # Read the first line to obtain the number of lines
        num_events = int(event_file.readline())
        events = np.zeros((num_events + 1, 2), dtype=np.int_)
        i = 1

        for line in event_file:
            tokens = line.strip().split(' ')
            events[i][0] = float(tokens[0])
            events[i][1] = ord(tokens[1])
            i += 1

        return events

