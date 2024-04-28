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
    page_title="",
    page_icon="📊",
    layout="centered",
    menu_items={
        "About": "# this is a header. This is a cool app!",
    },
)
st.markdown("# An Ordinal Classification Metric: Closeness Evaluation Measure")

st.markdown("## Introduction")
st.write("Conventionally, we typically ordinal classification tasks a")
st.write("Relative within the class itself")

st.markdown("## An Ordinal Classification Metrics")
st.write("Content for Section 1 goes here.")
st.write("The measure is not symmetrical.")
with st.form("paper_review_dist_form"):
    col1, col2 = st.columns(2, gap="small")
    with col1:
        st.markdown("### First distribution")
        first_review_confusion_matrix_df = pd.read_csv(
            "data/paper-review-example-01.csv", header=0
        )
        first_review_confusion_matrix_df = st.data_editor(
            first_review_confusion_matrix_df, key="example_dist_1"
        )

    with col2:
        st.markdown("### Second distribution")
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


st.markdown("## CEM definition")
st.write("Something")

# Section 2
st.markdown("## Metric Properties")
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
st.markdown("## Section 3")
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
