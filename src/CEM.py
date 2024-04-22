import math
from typing import List
import numpy as np
from src.ordinal_class_dist import OrdinalClassDistribution

SCALE_FACTOR = math.log10(2)

class ClosenessInformationQuantityCompute():
    """Compute the closeness information quantity (CIQ) for any class of 
    an ordinal class distribution.

    Args:
        ordinal_dist (OrdinalClassDistribution): an ordinal class distribution

    Attributes:
        ordinal_dist (OrdinalClassDistribution): an ordinal class distribution

    """
    def __init__(self, ordinal_dist: OrdinalClassDistribution) -> None:
        self.ordinal_dist = ordinal_dist

    def get_proximity_between_two_classes(self, first_class: str, second_class: str, is_log_enabled: bool=True) -> float:
        """Get the proximity between two classes.

        Args:
            first_class (str): a class name
            second_class (str): a class name

        Returns:
            float: proximity
        """
        ordinal_dist = self.ordinal_dist
        lower_class, higher_class = ordinal_dist.sort_two_classes_by_order(first_class, second_class)
        if (lower_class != higher_class):
            numerator = ordinal_dist.get_class_count(first_class)/2 +\
                ordinal_dist.get_sample_count_between_two_classes(lower_class, higher_class) + \
                ordinal_dist.get_class_count(second_class)
        else:
            numerator = ordinal_dist.get_class_count(first_class)/2
        
        denominator = ordinal_dist.get_total_count()
        if (denominator == 0): raise ZeroDivisionError('Each class must have at least one sample.')

        proximity = numerator / denominator
        if (is_log_enabled): proximity = -1*math.log10(proximity)/SCALE_FACTOR
        return proximity



class ClosenessEvaluationMeasureCompute():
    def __init__(
        self,
        confusion_matrix: np.array, 
        class_names: List[str],
        orders: List[int],
        ) -> None:
        self.confusion_matrix = confusion_matrix
        self.class_names = class_names
        self.orders = orders
        self.actual_ordinal_dist = OrdinalClassDistribution(
            class_names= self.class_names, 
            orders= self.orders,
            counts= self.compute_class_count_for_actual(),
            )
        self.CIQ_compute = ClosenessInformationQuantityCompute(self.actual_ordinal_dist)

    def get_proximity_between_two_dists(self) -> float:
        
        sum_numerator = 0
        sum_denominator = 0
        
        for actual_class in self.class_names:
            for predict_class in self.class_names:
                n_samples = self.confusion_matrix.loc[actual_class][predict_class]
                if (n_samples == 0): continue

                proximity_predict_vs_actual = self.CIQ_compute.get_proximity_between_two_classes(predict_class, actual_class)
                sum_numerator += n_samples * proximity_predict_vs_actual

                proximity_actual_vs_actual =  self.CIQ_compute.get_proximity_between_two_classes(actual_class, actual_class)
                sum_denominator += n_samples * proximity_actual_vs_actual

        proximity = sum_numerator / sum_denominator
        return proximity
    
    def get_proximity_matrix(self):
        proximity_matrix = 1. * self.confusion_matrix.copy()
        proximity_matrix.index.name = ''
        for first_actual_class in self.class_names:
            for second_actual_class in self.class_names:
                proximity_matrix.at[first_actual_class, second_actual_class] = self.CIQ_compute.get_proximity_between_two_classes(
                    first_class=first_actual_class, second_class=second_actual_class
                )
        
        return proximity_matrix

    def compute_class_count_for_actual(self) -> int:
        return self.confusion_matrix.sum(axis=1).tolist()