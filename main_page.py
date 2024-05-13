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

More formally, \n
"""
st.markdown(description, unsafe_allow_html=True)

with st.container(border=True):
    col1, col2 = st.columns(
        [2.5, 1],
    )

    with col1:
        st.markdown(
            """
            Given a set of ordinal classes<br><br><br>
            and amount of data points assigned to each class,<br><br><br>
            the informational closeness of any class pair <br><br>
            is the inverse <br><br><br>
            of the probability of data points assigned to classes in between, inclusively the two classes under consideration.
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.latex("C=\{c_1, ..., c_m\}")
        st.latex("N=\Sigma_{i=1}^{m} n_i")
        st.markdown("")
        st.latex("IC(c_i, c_j)")
        st.latex("-log(.)")
        st.markdown("")
        st.latex("\\frac{\\frac{n_i}{2} + \Sigma_{k=i+1}^{j} n_k}{N}")

st.markdown("Put it all together, we have")
st.latex("IC(c_i, c_j) = -log(\\frac{\\frac{n_i}{2} + \Sigma_{k=i+1}^{j} n_k}{N})")

st.markdown("To be clear,")
st.latex("IC(c_1, c_3) = -log(\\frac{n_1/2 + n_2 + n_3}{N})")
st.latex("IC(c_3, c_1) = -log(\\frac{n_3/2 + n_2 + n_1}{N})")
st.latex("IC(c_1, c_1) = -log(\\frac{n_1/2}{N})")

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
        st.markdown("### Journal $F$")
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
        st.markdown("### Journal $S$")
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

description = """
In the displayed charts for each journal, we computed $IC$ for all class combinations and show the top closest and furthest pairs.
As expected, both journals agree that *reject* vs. *accept* is the furthest pair.
Journal $S$ treats those on the border: *weakly reject*, *undecided*, *weakly accept* as closest pairs, given there are small amount of data points between them.

Note that, $IC$ is not symmetrical, hence $IC(c_i, c_j) \\neq IC(c_j, c_i)$. 
Also for visibility's sake, we scale the computed value by a constant of $1/log(2)$. 
"""
st.markdown(description)

st.markdown("## Closeness Evaluation Measure", unsafe_allow_html=True)
description = """
Next question: How to convey the idea of informational closeness to a usable evaluation metric for ordinal classification task?

It's simple. Firstly, you compute the $IC$ for classes in a particular dataset; 
this quantity translates to the reward for making an ordinal step from one class to another which are distributionally close.
Secondly, assume you have a candidate Machine Learning model has been trained on such dataset and ready to evaluate. 
For each data point, we favour prediction whose class is close to the groundtruth; 
an accurate class matching results to a point $1$, while the lesser earning lower point.
Doing the same for all predictions, we obtain the overall performance measure, i.e. $CEM$.
The highest achievable score for $CEM$ is $1$, that is the higher the better.

More formally,
"""
st.markdown(description)

with st.container(border=True):
    col1, col2 = st.columns(
        [2.5, 1],
    )

    with col1:
        st.markdown(
            """
            Given a dataset, <br><br><br>
            a groundtruth label mapping, <br><br><br>
            an ML model,<br><br>
            the closeness evaluation measure of such model<br><br>
            is the summation of $IC$ between predicted class and groundtruth, normalized by $IC$ of the groundtruth itself,<br><br>
            across all data points.
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.latex("D")
        st.latex("g: D \\rightarrow C")
        st.markdown("")
        st.latex("m: D \\rightarrow C")
        st.latex("CEM(m, g)")
        st.latex("\\frac{ IC[m(d), g(d)]} {IC[g(d), g(d)]}")
        st.latex("\Sigma_{d \in D}")

st.markdown("Put it all together, we have")
st.latex(
    "CEM(m, g) = \\frac{\Sigma_{d \in D} IC[m(d), g(d)]} {\Sigma_{d \in D} IC[g(d), g(d)]}"
)

st.markdown("## Metric Properties", unsafe_allow_html=True)
description = """
$CEM$ has been proven to satisfy three mathematical properties that address the shortcomings of other metrics.
Each property to be accompanied by a pair of datasets, whose groundtruth and prediction are depicted in the x-axis and coloring, respectively. 
This to illustrate the intuition behind each property, and in order to read the charts properly, 
do pay attention to Ordinal Invariance section.
"""
st.markdown(description)
st.markdown("### Ordinal Invariance")

description = """
**First Dataset** is a journal paper review dataset with 9 data points. 
The x-axis' ticks depict their true classes: $3$ *rejects*, $3$ *weak rejects*, $3$ *undecideds* and $0$ for the other two classes. 
The coloring of the circles ( 
<strong><span style='color: pink;'>pink</span></strong>, 
<strong><span style='color: red;'>red</span></strong>, 
<strong><span style='color: #FFD700;'>yellow</span></strong>, 
<strong><span style='color: #00FF00;'>dark green</span></strong>, 
<strong><span style='color: #98FB98;'>light green</span></strong>), i.e. model prediction, depicts the data points' predictions are exact matches to their true classes.

**Second Dataset** is similar with $100\%$ match for other set of classes: $3$ *rejects*, $3$ *undecideds*, $3$ *weak accepts*.

Both datasets result in the same metric value because both the model output and groundtruth shift their value in a strictly increasing manner.
"""
st.markdown(description, unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="small")
with col1:
    with st.container(border=True):

        st.markdown("### First Dataset")
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

        st.markdown("### Second Dataset")
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
description = """
Changing the model output closer to the groundtruth should result in a metric increase.

Indeed, **First Dataset** have one *weak accept* point misclassified as *undecided*.
**Second Dataset** should perform worse because the very same data point is misclassified to a further class, i.e. *weak reject*.
"""
st.markdown(description, unsafe_allow_html=True)
col1, col2 = st.columns(2, gap="small")
with col1:
    with st.container(border=True):

        st.markdown("### First Dataset")
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

        st.markdown("### Second Dataset")
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
description = """
A classification error in a small class should have lesser impact - or receive higher reward - in comparison to the same mistake in a frequent class.

**First Dataset** that misclassifies a data point from a frequent class *weak reject* should have lower metric value than 
**Second Dataset** that misclassifies a data point from a small class, i.e. *weak accept*.
"""
st.markdown(description, unsafe_allow_html=True)
st.warning(
    "There is an inconsistency in my interpretation and the original paper. I have contacted the author for advice regarding this discrepancy. [Raised Issue](https://github.com/EvALLTEAM/CEM-Ord/issues/1)"
)

col1, col2 = st.columns(2, gap="small")
with col1:
    with st.container(border=True):

        st.markdown("### First Dataset")
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

        st.markdown("### Second Dataset")
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
description = """
Let's put $CEM$ into an actual test!

Given a same dataset, we have the confusion matrix for two model $A$ and $B$ as followings.
From **accuracy** metric standpoint, both models are on par.
$CEM$ said otherwise and highlights that model $B$ should be the better performer.

It is indeed a true evaluation, in view that:
- Model $A$ makes more mistakes between distant classes *positive*-*negative* ($7+4 > 4+2$).
- Model $A$ makes more mistakes in *positive*-*neutral*, whose population represent $90\%$ of the dataset, hence penalized more heavily, or precisely, earning less reward.
"""
st.markdown(description)

with st.container(border=True):
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("### Model $A$'s Confusion Matrix")
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
        st.markdown("### Model $B$'s Confusion Matrix")
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
