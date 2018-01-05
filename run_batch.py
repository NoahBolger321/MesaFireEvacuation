from mesa.batchrunner import BatchRunner
import argparse
import matplotlib.pyplot as plt
import time
from datetime import timedelta
import sys
import os
import pandas as pd
import pickle

from fire_evacuation.model import FireEvacuation
from fire_evacuation.agent import Human

OUTPUT_DIR = "./output"

MIN_COLLABORATION = 0
MAX_COLLABORATION = 100

GRAPH_DPI = 100
GRAPH_WIDTH = 1920
GRAPH_HEIGHT = 1080


# Concatenate all of the dataframe files found in the OUTPUT_DIR
def merge_dataframes():
    directory = os.path.join(OUTPUT_DIR, "/batch_results/")

    previous_dataframe_files = [f for f in os.listdir(directory) if (os.path.isfile(os.path.join(directory, f)) and "dataframe_" in f)]

    # Concatenate any previous dataframes
    if previous_dataframe_files:
        dataframes = []
        print("Merging these dataframes:", previous_dataframe_files)

        for f in previous_dataframe_files:
            df = pickle.load(open(f, "rb"))
            dataframes.append(df)

        return pd.concat(dataframes, ignore_index=True), len(dataframes)  # Concatenate all of the dataframes together, while ignoring their indexes
    else:
        return None, None


parser = argparse.ArgumentParser()
parser.add_argument("runs", help="Number of repeat runs to do for each parameter setup.", type=int)
parser.add_argument("human_count", help="Number of humans in the simulation.", type=int)

try:
    args = parser.parse_args()
except Exception:
    parser.print_help()
    sys.exit(1)

runs = 1
if args.runs:
    runs = args.runs

human_count = 1
if args.human_count:
    human_count = args.human_count

# Fixed parameters of our batch runs
fixed_params = dict(floor_plan_file="floorplan_2.txt", human_count=human_count, fire_probability=0.8, visualise_vision=False, random_spawn=True, save_plots=True)

# Vary percentage collaboration between MIN and MAX values above
collaboration_range = range(MIN_COLLABORATION, MAX_COLLABORATION + 1, 10)
variable_params = dict(collaboration_percentage=collaboration_range)

# At the end of each model run, calculate the percentage of people that escaped
model_reporter = {"PercentageEscaped": lambda m: ((FireEvacuation.count_human_status(m, Human.Status.ESCAPED) / m.human_count) * 100)}

# Create the batch runner
print("Running batch test with %i runs for each parameter and %i human agents." % (runs, human_count))

start = time.time()  # Time the batch run
for i in range(1, runs + 1):
    param_run = BatchRunner(FireEvacuation, variable_parameters=variable_params, fixed_parameters=fixed_params, model_reporters=model_reporter)

    param_run.run_all()  # Run all simulations

    end = time.time()
    end_timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save the dataframe to a file so we have the oppurtunity to concatenate separate dataframes from separate runs
    dataframe = param_run.get_model_vars_dataframe()
    dataframe.to_pickle(path=os.path.join(OUTPUT_DIR, "/batch_results/dataframe_" + end_timestamp + ".pickle"))

    elapsed = end - start  # Get the elapsed time in seconds
    print("Batch runner finished iteration %i. Took: %s" % (i, str(timedelta(seconds=elapsed))))

dataframe, count = merge_dataframes()
del dataframe['Run']
dataframe.groupby("collaboration_percentage")

fig = plt.figure(figsize=(GRAPH_WIDTH / GRAPH_DPI, GRAPH_HEIGHT / GRAPH_DPI), dpi=GRAPH_DPI)
plt.scatter(dataframe.collaboration_percentage, dataframe.PercentageEscaped)
fig.suptitle("Evacuation Success: " + str(human_count) + " Human Agents, " + str(count) + " Iterations", fontsize=20)
plt.xlabel("Percentage of Humans Collaborating (%)", fontsize=14)
plt.ylabel("Percentage Escaped (%)", fontsize=14)
plt.xticks(range(MIN_COLLABORATION, MAX_COLLABORATION + 1, 10))
plt.ylim(0, 100)
plt.savefig(os.path.join(OUTPUT_DIR, "/batch_graphs/batch_run_scatter_" + end_timestamp + ".png"), dpi=GRAPH_DPI)

fig = plt.figure(figsize=(GRAPH_WIDTH / GRAPH_DPI, GRAPH_HEIGHT / GRAPH_DPI), dpi=GRAPH_DPI)
ax = fig.gca()

# TODO: lookup why it plots by index...
dataframe.boxplot(ax=ax, column="PercentageEscaped", by="collaboration_percentage", positions=list(collaboration_range))
fig.suptitle("Evacuation Success: " + str(human_count) + " Human Agents, " + str(count) + " Iterations", fontsize=20)
plt.xlabel("Percentage of Humans Collaborating (%)", fontsize=14)
plt.ylabel("Percentage Escaped (%)", fontsize=14)
plt.xticks(collaboration_range)
plt.ylim(0, 100)
plt.savefig(os.path.join(OUTPUT_DIR, "/batch_graphs/batch_run_boxplot_" + end_timestamp + ".png"), dpi=GRAPH_DPI)

print(dataframe)
