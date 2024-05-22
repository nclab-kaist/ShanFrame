import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from ..ir import Model
from ..utils import get_rect


def visualize_memory(model: Model, peak_mem = 0, scale=1024, path = "footprint.png"):
    plt.ioff()
    fig, ax = plt.subplots()
    rects = get_rect(model)

    # add each rect to plot
    for rect in rects:
        height = rect.height
        width = rect.width
        x = rect.start
        y = rect.addr

        # create and add a patch
        patch = patches.Rectangle(
            (x, y), width, height, edgecolor='black', facecolor='yellow')
        ax.add_patch(patch)

    x_axis_max = max([rect.start + rect.width for rect in rects]) + 1
    y_axis_max = max([rect.addr + rect.height for rect in rects]) + 1
    if y_axis_max < peak_mem:
        y_axis_max = peak_mem
    ax.set_xlim(0, x_axis_max)
    ax.set_ylim(0, y_axis_max)

    # set titles
    fig.suptitle("Memory footprint per layer during inference")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Memory footprint (Byte)")

    plt.savefig(path)
    plt.close()
