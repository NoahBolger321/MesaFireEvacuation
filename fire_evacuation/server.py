import os
import cv2
import numpy as np
from os import listdir, path
from enum import IntEnum
from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import UserSettableParameter

from .model import FireEvacuation
from .agent import FireExit, Wall, Furniture, Fire, Smoke, Human, Sight, Door, DeadHuman


def get_door_size(floorplan):
    # create a zeros array same size as the txt floorplan
    floorplan_labels = np.zeros(floorplan.shape[:2], dtype=np.uint8)
    # wherever we have an "E" (door), set the value equal to 1
    floorplan_labels[np.where(floorplan == "E")] = 1
    # convert to numpy uint8
    np.uint8(floorplan_labels)

    # find all of the connected components
    output = cv2.connectedComponentsWithStats(floorplan_labels, 4, cv2.CV_32S)
    # The third cell is the stat matrix
    stats = output[2]
    # get all of the areas (except for the first area which is the empty space component)
    areas = list(stats[1:, -1])
    # to get door shape, average all door areas and take square root (makes assumption that doors are roughly square)
    return np.sqrt(np.mean(areas))

# Get list of available floorplans
floor_plans = [
    f
    for f in listdir("fire_evacuation/floorplans")
    if path.isfile(path.join("fire_evacuation/floorplans", f))
]

# get the floorplan dimensions to set the grid dimensions
with open(os.path.join("fire_evacuation/floorplans/", floor_plans[0]), "rt") as f:
    floorplan = np.matrix([line.strip().split() for line in f.readlines()])

# Rotate the floorplan so it's interpreted as seen in the text file
floorplan = np.rot90(floorplan, 3)
height, width = np.shape(floorplan)

DOOR_SIZE = get_door_size(floorplan)

# scale Human agent mobility and max speed to reflect size of doors
class Mobility(IntEnum):
    INCAPACITATED = 0 * DOOR_SIZE
    NORMAL = 0 * DOOR_SIZE
    PANIC = 0 * DOOR_SIZE
Human.Mobility = Mobility
Human.MAX_SPEED = 2.0 * DOOR_SIZE
FireEvacuation.MAX_SPEED = int(2 * DOOR_SIZE)
Fire.smoke_radius = int(DOOR_SIZE)
Smoke.smoke_radius = int(DOOR_SIZE)
Smoke.spread_rate = int(DOOR_SIZE // 4)
Smoke.spread_threshold = 5*int(DOOR_SIZE)


# Creates a visual portrayal of our model in the browser interface
def fire_evacuation_portrayal(agent):
    if agent is None:
        return

    portrayal = {}
    (x, y) = agent.get_position()
    portrayal["x"] = x
    portrayal["y"] = y

    if type(agent) is Human:
        portrayal["scale"] = DOOR_SIZE
        portrayal["Layer"] = 5

        if agent.get_mobility() == Human.Mobility.INCAPACITATED:
            # Incapacitated
            portrayal["Shape"] = "fire_evacuation/resources/incapacitated_human.png"
            portrayal["Layer"] = 6
        elif agent.get_mobility() == Human.Mobility.PANIC:
            # Panicked
            portrayal["Shape"] = "fire_evacuation/resources/panicked_human.png"
        elif agent.is_carrying():
            # Carrying someone
            portrayal["Shape"] = "fire_evacuation/resources/carrying_human.png"
        else:
            # Normal
            portrayal["Shape"] = "fire_evacuation/resources/human.png"
    elif type(agent) is Fire:
        portrayal["Shape"] = "fire_evacuation/resources/fire.png"
        portrayal["scale"] = DOOR_SIZE // 2
        portrayal["Layer"] = 3
    elif type(agent) is Smoke:
        portrayal["Shape"] = "fire_evacuation/resources/smoke.png"
        portrayal["scale"] = DOOR_SIZE // 2
        portrayal["Layer"] = 2
    elif type(agent) is FireExit:
        portrayal["Shape"] = "fire_evacuation/resources/fire_exit.png"
        portrayal["scale"] = 1
        portrayal["Layer"] = 1
    elif type(agent) is Door:
        portrayal["Shape"] = "fire_evacuation/resources/door.png"
        portrayal["scale"] = 1
        portrayal["Layer"] = 1
    elif type(agent) is Wall:
        portrayal["Shape"] = "fire_evacuation/resources/wall.png"
        portrayal["scale"] = 1
        portrayal["Layer"] = 1
    elif type(agent) is Furniture:
        portrayal["Shape"] = "fire_evacuation/resources/furniture.png"
        portrayal["scale"] = 1
        portrayal["Layer"] = 1
    elif type(agent) is DeadHuman:
        portrayal["Shape"] = "fire_evacuation/resources/dead.png"
        portrayal["scale"] = DOOR_SIZE
        portrayal["Layer"] = 4
    elif type(agent) is Sight:
        portrayal["Shape"] = "fire_evacuation/resources/eye.png"
        portrayal["scale"] = 0.8
        portrayal["Layer"] = 7

    return portrayal

# Define the charts on our web interface visualisation
status_chart = ChartModule(
    [
        {"Label": "Alive", "Color": "blue"},
        {"Label": "Dead", "Color": "red"},
        {"Label": "Escaped", "Color": "green"},
    ]
)

mobility_chart = ChartModule(
    [
        {"Label": "Normal", "Color": "green"},
        {"Label": "Panic", "Color": "red"},
        {"Label": "Incapacitated", "Color": "blue"},
    ]
)

collaboration_chart = ChartModule(
    [
        {"Label": "Verbal Collaboration", "Color": "orange"},
        {"Label": "Physical Collaboration", "Color": "red"},
        {"Label": "Morale Collaboration", "Color": "pink"},
    ]
)


canvas_element = CanvasGrid(fire_evacuation_portrayal, height, width, 800, 800)
f.close()

# Specify the parameters changeable by the user, in the web interface
model_params = {
    "floor_plan_file": UserSettableParameter(
        "choice", "Floorplan", value=floor_plans[0], choices=floor_plans
    ),
    "human_count": UserSettableParameter("number", "Number Of Human Agents", value=10),
    "collaboration_percentage": UserSettableParameter(
        "slider", "Percentage Collaborating", value=50, min_value=0, max_value=100, step=10
    ),
    "fire_probability": UserSettableParameter(
        "slider", "Probability of Fire", value=0.1, min_value=0, max_value=1, step=0.01
    ),
    "random_spawn": UserSettableParameter(
        "checkbox", "Spawn Agents at Random Locations", value=True
    ),
    "visualise_vision": UserSettableParameter("checkbox", "Show Agent Vision", value=False),
    "save_plots": UserSettableParameter("checkbox", "Save plots to file", value=True),
}

# Start the visual server with the model
server = ModularServer(
    FireEvacuation,
    [canvas_element, status_chart, mobility_chart, collaboration_chart],
    "Fire Evacuation",
    model_params,
)
