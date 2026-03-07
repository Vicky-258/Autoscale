def violation_frequency(window):
    return sum(window)


def max_run_length(window):
    max_run = 0
    current = 0

    for v in window:
        if v == 1:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 0

    return max_run


def run_count(window):
    runs = 0
    in_run = False

    for v in window:
        if v == 1 and not in_run:
            runs += 1
            in_run = True
        elif v == 0:
            in_run = False

    return runs
