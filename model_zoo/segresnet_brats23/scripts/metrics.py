from typing import Callable

from monai.handlers.ignite_metric import IgniteMetricHandler
from monai.metrics import DiceMetric, HausdorffDistanceMetric


class MeanDice(IgniteMetricHandler):
    """
    Extends MONIA `MeanDice` class, which computes Dice score metric from full size Tensor and collects average over batch, class-channels, iterations.
    Most of the code is taken from the MONAI's original implemetation of `MeanDice`.
    """

    def __init__(
        self,
        output_transform: Callable = lambda x: x,
        save_details: bool = True,
        **kwargs
    ) -> None:

        metric_fn = DiceMetric(**kwargs)
        super().__init__(
            metric_fn=metric_fn,
            output_transform=output_transform,
            save_details=save_details,
        )


class MeanHausdorffDistance(IgniteMetricHandler):
    """
    Computes Hausdorff distance from full size Tensor and collects average over batch, class-channels, iterations.
    """

    def __init__(
        self,
        output_transform: Callable = lambda x: x,
        save_details: bool = True,
        **kwargs
    ) -> None:

        metric_fn = HausdorffDistanceMetric(**kwargs)
        super().__init__(
            metric_fn=metric_fn,
            output_transform=output_transform,
            save_details=save_details,
        )
