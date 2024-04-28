from enum import Enum
from typing import Dict
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np


from src.CEM import ClosenessInformationQuantityCompute
from src.ordinal_class_dist import OrdinalClassDistribution


class RGBColor(Enum):
    RED = (1, 0, 0)
    LIGHT_RED = (1, 0.5, 0.5)
    LIGHT_ORANGE = (1, 0.9, 0.5)
    LIGHT_GREEN = (0.5, 1, 0.5)
    GREEN = (0, 1, 0)


FIVE_COLOR_LIST = [
    color.value
    for color in [
        RGBColor.LIGHT_RED,
        RGBColor.RED,
        RGBColor.LIGHT_ORANGE,
        RGBColor.GREEN,
        RGBColor.LIGHT_GREEN,
    ]
]

THREE_COLOR_LIST = [
    color.value
    for color in [
        # RGBColor.LIGHT_RED,
        RGBColor.RED,
        RGBColor.LIGHT_ORANGE,
        RGBColor.GREEN,
        # RGBColor.LIGHT_GREEN,
    ]
]


def scatter_plot_from_single_class_distribution(class_dist_df: pd.DataFrame):
    data = list()

    # generate records based on count in confusion matrix
    for _, row in class_dist_df.iterrows():
        class_name, order, count = row.tolist()
        data.extend([[class_name, order]] * count)

    scatter_data = pd.DataFrame(data=data, columns=["class_name", "order"])
    numb_points = scatter_data.shape[0]
    noise_x = np.random.uniform(-0.3, 0.3, size=numb_points)
    noise_y = np.random.uniform(-0.8, 0.8, size=numb_points)

    class_x_mapping = dict(zip(class_dist_df["class_name"], class_dist_df["order"]))
    scatter_data["x"] = (
        scatter_data["class_name"].apply(lambda x: class_x_mapping[x]) + noise_x
    )
    scatter_data["y"] = 1 + noise_y

    numb_classes = len(class_dist_df["class_name"])
    if numb_classes == 3:
        class_color_mapping = dict(zip(class_dist_df["class_name"], THREE_COLOR_LIST))
    elif numb_classes == 5:
        class_color_mapping = dict(zip(class_dist_df["class_name"], FIVE_COLOR_LIST))
    scatter_data["color"] = scatter_data["class_name"].apply(
        lambda x: class_color_mapping[x]
    )

    fig, ax = plt.subplots()
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 2.5)
    ax.set_xticks(class_dist_df["order"])
    ax.set_xticklabels(class_dist_df["class_name"], rotation=30)
    ax.set_yticks([])
    ax.scatter(
        scatter_data["x"],
        scatter_data["y"],
        c=scatter_data["color"],
        alpha=0.8,
        edgecolor="black",
        s=70,
    )

    # assign total count per class
    for count, order in zip(class_dist_df["count"], class_dist_df["order"]):
        ax.text(order, 2.2, str(count), ha="center")

    return fig


def scatter_plot_from_confusion_matrix(confusion_matrix_df: pd.DataFrame):
    class_names = confusion_matrix_df.columns.tolist()[1:]
    confusion_matrix_query_df = confusion_matrix_df.set_index("actual\predict")
    count_data = list()
    for actual_class in class_names:
        for predict_class in class_names:
            count = confusion_matrix_query_df.loc[actual_class][predict_class]
            count_data.append([actual_class, predict_class, count])

    scatter_data = list()
    numb_classes = len(class_names)
    class_orders = list(range(1, numb_classes + 1))
    actual_class_x_mapping = dict(zip(class_names, class_orders))
    if numb_classes == 3:
        predict_class_color_mapping = dict(zip(class_names, THREE_COLOR_LIST))
    elif numb_classes == 5:
        predict_class_color_mapping = dict(zip(class_names, FIVE_COLOR_LIST))
    for actual_class, predict_class, count in count_data:
        if count == 0:
            continue

        x_val = actual_class_x_mapping[actual_class]
        color = predict_class_color_mapping[predict_class]
        scatter_data.extend([[x_val, color]] * count)

    scatter_data = pd.DataFrame(scatter_data, columns=["x", "color"])
    numb_points = scatter_data.shape[0]
    noise_x = np.random.uniform(-0.15, 0.15, size=numb_points)
    noise_y = np.random.uniform(-0.8, 0.8, size=numb_points)
    scatter_data["x"] += noise_x
    scatter_data["y"] = 1 + noise_y

    fig, ax = plt.subplots()
    ax.set_xlim(0, numb_classes + 1)
    ax.set_ylim(0, 2.5)
    ax.set_xticks(class_orders)
    ax.set_xticklabels(class_names, rotation=30)
    ax.set_yticks([])
    ax.scatter(
        scatter_data["x"],
        scatter_data["y"],
        c=scatter_data["color"],
        alpha=0.8,
        edgecolor="black",
        s=70,
    )

    return fig


def get_class_proximity_dict_from_confusion_matrix(confusion_matrix_df: pd.DataFrame):
    dist = OrdinalClassDistribution(
        confusion_matrix_df["class_name"],
        confusion_matrix_df["order"],
        confusion_matrix_df["count"],
    )
    CIQ_compute = ClosenessInformationQuantityCompute(dist)

    class_proximity_dict = dict()
    for first_class_name in confusion_matrix_df["class_name"]:
        for second_class_name in confusion_matrix_df["class_name"]:
            if first_class_name == second_class_name:
                continue
            proximity = CIQ_compute.get_proximity_between_two_classes(
                first_class_name, second_class_name
            )
            class_proximity_dict[f"{first_class_name}-{second_class_name}"] = proximity

    return class_proximity_dict


def get_top_n_from_dict(data_dict: Dict, n: int, descending=True):
    sorted_data_dict = dict(
        sorted(data_dict.items(), key=lambda x: x[1], reverse=descending)
    )
    sorted_data_list = list(sorted_data_dict.items())
    return sorted_data_list[:n]
