import json

def init_mpl():
    import matplotlib
    import matplotlib.pyplot as plt
    SMALL_SIZE = 16
    TICK_SIZE = 19
    MEDIUM_SIZE = 22
    BIGGER_SIZE = 24
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=TICK_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=TICK_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the figure titl

    return plt


def load_jsonl(path):
    with open(path, 'r') as f:
        rows = [json.loads(line) for line in f]

    # convert to column format
    columns = {}
    for row in rows:
        for key, value in row.items():
            if key not in columns:
                columns[key] = []
            columns[key].append(value)

    return columns


def ema(x, alpha=0.15):
    if len(x) == 0:
        return x
    
    y = [x[0]]
    for i in range(1, len(x)):
        y.append(alpha * y[-1] + (1 - alpha) * x[i])
    
    return y
