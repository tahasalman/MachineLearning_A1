from random import random
ACTIVITIES_COUNT = {
    "M": 0,
    "C": 0,
    "P": 0,
    "S": 0
}

def pick_activity():
    random_num = random()
    if random_num < 0.2:
        return "M"
    elif random_num < 0.6:
        return "C"
    elif random_num < 0.7:
        return "P"
    else:
        return "S"

def simulate_days(num_days):
    count = 0
    while count<num_days:
        activity = pick_activity()
        ACTIVITIES_COUNT[activity] += 1
        count+=1

    distribution = {}

    for activity in ACTIVITIES_COUNT:
        distribution[activity] = ACTIVITIES_COUNT[activity]/num_days

    return distribution

if __name__=="__main__":
    SAMPLE_DISTRIBUTION = simulate_days(1000)
    from pprint import pprint
    pprint(ACTIVITIES_COUNT)
    pprint(SAMPLE_DISTRIBUTION)
