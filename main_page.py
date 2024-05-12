import streamlit as st
import numpy as np

np.random.seed(42)
import pandas as pd

from src.CEM import ClosenessEvaluationMeasureCompute
from src.utils import compute_accuracy_score

from src.viz_utils import (
    get_class_proximity_dict_from_confusion_matrix,
    get_top_n_from_dict,
    scatter_plot_from_confusion_matrix,
    scatter_plot_from_single_class_distribution,
)

st.set_page_config(
    page_title="Closeness Evaluation Measure",
    page_icon="📊",
    layout="centered",
    menu_items={
        "About": "# this is a header. This is a cool app!",
    },
)

st.markdown("# An Ordinal Classification Metric: Closeness Evaluation Measure")
st.markdown(
    """
    [Original Blog](url) |
    [Github](https://github.com/hovinh/closeness-evaluation-measure) |
    [Cited Paper](https://aclanthology.org/2020.acl-main.363.pdf)
    """
)

st.markdown("## Introduction")
description = """
**Ordinal classification** is a type of classification task whose predicted classes (or categories) have a specific ordering.
This implies misclassifying a class pair should be penalized differently and proportionally, with respect to their semantic order and class distribution in the dataset.
Existing evaluation metrics are shown failed to capture certain aspects of this task, that it calls for the employment of a new metric, namely **Closeness Evaluation Measure** (**$CEM$**).

To be clear,
- precision/recall ignores the ordering among classes.
- mean average error/mean squared error assumes a predefined interval between classes, subject to their numeric encoding.
- ranking measures indeed capture the relative ranking in terms of relevancy, but lack of emphasis on identifying the correct class.

$CEM$ employs the idea of informational closeness: the more **UNLIKELY** there is data point between a class pair, the closer they are.
As a result, a model that misclassifies informational-close classes should be penalized lesser than those informational-distant.
It has been mathematically proven this measure attains the desired properties that addresses the foregoing shortcomings, which to be discussed shortly after.

This demo is divided as follows:
- [Informational Closeness](#informational-closeness)
- [Closeness Evaluation Measure](#closeness-evaluation-measure)
- [Metric Properties](#metric-properties)
- [An Example Usecase](#an-example-usecase)
"""
st.markdown(description)

st.markdown("## Informational Closeness", unsafe_allow_html=True)
description = """
Let's say we have two journals $F$ and $S$ (mockup data below) with two very distinct paper review methods.
Reviewers at journal $F$ are very cautious and it's rare to see they review a paper as hard *reject* or hard *accept*.
Journal $S$, on the other hand, tends to take a clear stance on the paper's acceptance/rejection status.

Given that, *weak reject* vs. *weak accept* in the context of $S$ are informational close because they are closer assessment, while
$F$ treating them as two far ends of the grading scales due to the fact this is a strong disagreement between reviewers in the context of $F$.
Put it differently, two classes $a$ and $b$ are informationally close if the probability of finding data points between the two is low.

More formally,
"""
st.markdown(description)

html_code = """
<div style="display: flex;">
    <div style="flex: 1; border-right: 1px solid black; padding-right: 10px; text-align: center;">
        Lorem ipsum dolor sit amet, consectetur adipiscing elit. <br>Sed euismod, sem vel dapibus aliquam, velit nisi dictum mi, vel bibendum nisl nulla non leo.
    </div>
    <div style="flex: 1; padding-left: 10px;text-align: center;">
        Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed euismod, sem vel dapibus aliquam, velit nisi dictum mi, vel bibendum nisl nulla non leo.
    </div>
</div>
"""
st.markdown(html_code, unsafe_allow_html=True)

description = """
Formally, 
$F$'s   
Taking the 
The measure is not symmetrical.
Formula
"""
st.markdown(description)

with st.form("paper_review_dist_form"):
    col1, col2 = st.columns(2, gap="small")
    with col1:
        st.markdown("### Journal $F$")
        first_review_confusion_matrix_df = pd.read_csv(
            "data/paper-review-example-01.csv", header=0
        )
        first_review_confusion_matrix_df = st.data_editor(
            first_review_confusion_matrix_df, key="example_dist_1"
        )

    with col2:
        st.markdown("### Journal $S$")
        second_review_confusion_matrix_df = pd.read_csv(
            "data/paper-review-example-02.csv", header=0
        )
        second_review_confusion_matrix_df = st.data_editor(
            second_review_confusion_matrix_df, key="example_dist_2"
        )

    st.form_submit_button("Compute")

col1, col2 = st.columns(2, gap="small")
with col1:
    with st.container(border=True):
        fig = scatter_plot_from_single_class_distribution(
            first_review_confusion_matrix_df
        )
        st.pyplot(fig)

        class_proximity_dict = get_class_proximity_dict_from_confusion_matrix(
            first_review_confusion_matrix_df
        )

        top_col1, top_col2 = st.columns(2)
        with top_col1:
            with st.container(border=True):
                st.markdown("Top Closest")
                top_closest = get_top_n_from_dict(
                    class_proximity_dict, n=3, descending=True
                )
                for class_pair, proximity in top_closest:
                    st.metric(
                        label=class_pair, value=f"{proximity:.3f}", delta_color="off"
                    )

        with top_col2:
            with st.container(border=True):
                st.markdown("Top Furthest")
                top_furthest = get_top_n_from_dict(
                    class_proximity_dict, n=3, descending=False
                )
                for class_pair, proximity in top_furthest:
                    st.metric(
                        label=class_pair, value=f"{proximity:.3f}", delta_color="off"
                    )

with col2:
    with st.container(border=True):
        fig = scatter_plot_from_single_class_distribution(
            second_review_confusion_matrix_df
        )
        st.pyplot(fig)

        class_proximity_dict = get_class_proximity_dict_from_confusion_matrix(
            second_review_confusion_matrix_df
        )

        top_col1, top_col2 = st.columns(2)
        with top_col1:
            with st.container(border=True):
                st.markdown("Top Closest")
                top_closest = get_top_n_from_dict(
                    class_proximity_dict, n=3, descending=True
                )
                for class_pair, proximity in top_closest:
                    st.metric(
                        label=class_pair, value=f"{proximity:.3f}", delta_color="off"
                    )

        with top_col2:
            with st.container(border=True):
                st.markdown("Top Furthest")
                top_furthest = get_top_n_from_dict(
                    class_proximity_dict, n=3, descending=False
                )
                for class_pair, proximity in top_furthest:
                    st.metric(
                        label=class_pair, value=f"{proximity:.3f}", delta_color="off"
                    )


st.markdown("## Closeness Evaluation Measure", unsafe_allow_html=True)
st.write("Something")

# Section 2
st.markdown("## Metric Properties", unsafe_allow_html=True)
st.write("Content for Section 2 goes here.")
# Showing characteristics of metrics
st.markdown("### Ordinal Invariance")

col1, col2 = st.columns(2, gap="small")
with col1:
    with st.container(border=True):

        st.markdown("### First distribution")
        csv_path = "data/ordinal_invariance-example-01.csv"
        first_ordinal_variance_df = pd.read_csv(csv_path, header=0)
        fig = scatter_plot_from_confusion_matrix(first_ordinal_variance_df)
        st.pyplot(fig)
        CEM_compute = ClosenessEvaluationMeasureCompute(
            confusion_matrix=first_ordinal_variance_df.set_index("actual\predict"),
            class_names=["reject", "weak reject", "undecided", "weak accept", "accept"],
            orders=[1, 2, 3, 4, 5],
        )

        CEM_value_1 = CEM_compute.get_proximity_between_two_dists()
        st.metric(label="CEM score", value=f"{CEM_value_1:.3f}")

with col2:
    with st.container(border=True):

        st.markdown("### Second distribution")
        csv_path = "data/ordinal_invariance-example-02.csv"
        first_ordinal_variance_df = pd.read_csv(csv_path, header=0)
        fig = scatter_plot_from_confusion_matrix(first_ordinal_variance_df)
        st.pyplot(fig)
        CEM_compute = ClosenessEvaluationMeasureCompute(
            confusion_matrix=first_ordinal_variance_df.set_index("actual\predict"),
            class_names=["reject", "weak reject", "undecided", "weak accept", "accept"],
            orders=[1, 2, 3, 4, 5],
        )

        CEM_value_2 = CEM_compute.get_proximity_between_two_dists()
        st.metric(label="CEM score", value=f"{CEM_value_2:.3f}")

st.markdown("### Monotonicity")
col1, col2 = st.columns(2, gap="small")
with col1:
    with st.container(border=True):

        st.markdown("### First distribution")
        csv_path = "data/monotonicity-example-01.csv"
        first_ordinal_variance_df = pd.read_csv(csv_path, header=0)
        fig = scatter_plot_from_confusion_matrix(first_ordinal_variance_df)
        st.pyplot(fig)
        CEM_compute = ClosenessEvaluationMeasureCompute(
            confusion_matrix=first_ordinal_variance_df.set_index("actual\predict"),
            class_names=["reject", "weak reject", "undecided", "weak accept", "accept"],
            orders=[1, 2, 3, 4, 5],
        )

        CEM_value_1 = CEM_compute.get_proximity_between_two_dists()
        st.metric(label="CEM score", value=f"{CEM_value_1:.3f}")

with col2:
    with st.container(border=True):

        st.markdown("### Second distribution")
        csv_path = "data/monotonicity-example-02.csv"
        first_ordinal_variance_df = pd.read_csv(csv_path, header=0)
        fig = scatter_plot_from_confusion_matrix(first_ordinal_variance_df)
        st.pyplot(fig)
        CEM_compute = ClosenessEvaluationMeasureCompute(
            confusion_matrix=first_ordinal_variance_df.set_index("actual\predict"),
            class_names=["reject", "weak reject", "undecided", "weak accept", "accept"],
            orders=[1, 2, 3, 4, 5],
        )

        CEM_value_2 = CEM_compute.get_proximity_between_two_dists()
        st.metric(
            label="CEM score",
            value=f"{CEM_value_2:.3f}",
            delta=f"{CEM_value_2-CEM_value_1:.3f}",
        )


st.markdown("### Imbalance")
col1, col2 = st.columns(2, gap="small")
with col1:
    with st.container(border=True):

        st.markdown("### First distribution")
        csv_path = "data/imbalance-example-01.csv"
        first_ordinal_variance_df = pd.read_csv(csv_path, header=0)
        fig = scatter_plot_from_confusion_matrix(first_ordinal_variance_df)
        st.pyplot(fig)
        CEM_compute = ClosenessEvaluationMeasureCompute(
            confusion_matrix=first_ordinal_variance_df.set_index("actual\predict"),
            class_names=["reject", "weak reject", "undecided", "weak accept", "accept"],
            orders=[1, 2, 3, 4, 5],
        )

        CEM_value_1 = CEM_compute.get_proximity_between_two_dists()
        st.metric(label="CEM score", value=f"{CEM_value_1:.3f}")

with col2:
    with st.container(border=True):

        st.markdown("### Second distribution")
        csv_path = "data/imbalance-example-02.csv"
        first_ordinal_variance_df = pd.read_csv(csv_path, header=0)
        fig = scatter_plot_from_confusion_matrix(first_ordinal_variance_df)
        st.pyplot(fig)
        CEM_compute = ClosenessEvaluationMeasureCompute(
            confusion_matrix=first_ordinal_variance_df.set_index("actual\predict"),
            class_names=["reject", "weak reject", "undecided", "weak accept", "accept"],
            orders=[1, 2, 3, 4, 5],
        )

        CEM_value_2 = CEM_compute.get_proximity_between_two_dists()
        st.metric(
            label="CEM score",
            value=f"{CEM_value_2:.3f}",
            delta=f"{CEM_value_2-CEM_value_1:.3f}",
        )

# Section 3
st.markdown("## An Example Usecase", unsafe_allow_html=True)

# showing the comparisoin between 2 distribution with the same accuracy

with st.container(border=True):
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("### Model A's Confusion Matrix")
        systemA_df = pd.read_csv("data/systemA-confusion-matrix.csv", header=0)
        systemA_df
    with col2:
        CEM_compute = ClosenessEvaluationMeasureCompute(
            confusion_matrix=systemA_df.set_index("actual\predict"),
            class_names=["negative", "neutral", "positive"],
            orders=[1, 2, 3],
        )
        accuracy_1 = compute_accuracy_score(
            systemA_df.set_index("actual\predict").values
        )
        CEM_value_1 = CEM_compute.get_proximity_between_two_dists()
        st.write("")
        st.write("")
        st.metric(label="Accuracy", value=f"{accuracy_1:.3f}")
        st.metric(label="CEM score", value=f"{CEM_value_1:.3f}")

with st.container(border=True):
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("### Model B's Confusion Matrix")
        systemB_df = pd.read_csv("data/systemB-confusion-matrix.csv", header=0)
        systemB_df
    with col2:
        CEM_compute = ClosenessEvaluationMeasureCompute(
            confusion_matrix=systemB_df.set_index("actual\predict"),
            class_names=["negative", "neutral", "positive"],
            orders=[1, 2, 3],
        )
        accuracy_2 = compute_accuracy_score(
            systemB_df.set_index("actual\predict").values
        )
        CEM_value_2 = CEM_compute.get_proximity_between_two_dists()
        st.write("")
        st.write("")
        st.metric(label="Accuracy", value=f"{accuracy_2:.3f}")
        st.metric(
            label="CEM score",
            value=f"{CEM_value_2:.3f}",
            delta=f"{CEM_value_2-CEM_value_1:.3f}",
        )


proximity_matrix = CEM_compute.get_proximity_matrix()
proximity_matrix
