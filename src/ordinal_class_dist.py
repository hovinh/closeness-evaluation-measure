from typing import List, Tuple

class OrdinalClassDistribution:
    """A class to encapsulate the distribution of classes in an ordinal classification.

    Args:
        class_names: List of class names
        orders: List of class orders
        counts: List of class counts
    Attributes:
        class_names: List of class names
        orders: List of class orders
        counts: List of class counts
        class_order_mapping: Mapping from class name to order
        order_class_mapping: Mapping from order to class name
        class_frequency_mapping: Mapping from class name to frequency

    """
    def __init__(self, class_names: List, orders: List, counts: List) -> None:
        self.class_names = class_names
        self.orders = sorted(orders)
        self.counts = counts
        self.class_order_mapping = dict(zip(class_names, orders))
        self.order_class_mapping = dict(zip(orders, class_names))
        self.class_frequency_mapping = dict(zip(class_names, counts))

    def get_class_count(self, class_name: str) -> int:
        return self.class_frequency_mapping.get(class_name, 0)
    
    def get_sample_count_between_two_classes(self, first_class: str, second_class: str) -> int:
        """Get the sample count between two classes.

        Args:
            first_class (str): a class name
            second_class (str): a class name

        Returns:
            int: count of sample between two classes
        """
        lower_class, higher_class = self.sort_two_classes_by_order(first_class, second_class)
        in_between_classes = self.get_class_names_in_between(lower_class, higher_class)

        sample_count = 0
        for class_name in in_between_classes:
            sample_count += self.get_class_count(class_name)
        return sample_count
    
    def get_total_count(self) -> int:
        return sum(self.counts)

    def sort_two_classes_by_order(self, first_class: str, second_class: str) -> Tuple[str, str]:
        """Given two classes, return them in sorted order.

        Args:
            first_class (str): a class name
            second_class (str): a class name

        Returns:
            Tuple[str, str]: sorted class names
        """
        first_class_order = self.class_order_mapping.get(first_class, 0)
        second_class_order = self.class_order_mapping.get(second_class, 0)
        if first_class_order < second_class_order:
            return first_class, second_class
        else:
            return second_class, first_class

    def get_class_names_in_between(self, lower_class: str, higher_class: str) -> List[str]:
        """Given two classes, return all class names in between (exclusively).

        Args:
            lower_class (str): a class name
            higher_class (str): a class name

        Returns:
            List[str]: list of class names in between
        """
        lower_class_order = self.class_order_mapping.get(lower_class, 0)
        higher_class_order = self.class_order_mapping.get(higher_class, 0)
        in_between_class_names = list()
        for order in self.orders:
            if (lower_class_order < order < higher_class_order):
                in_between_class_names.append(self.order_class_mapping[order])
        return in_between_class_names