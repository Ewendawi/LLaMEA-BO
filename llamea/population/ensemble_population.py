
from enum import Enum
from typing import Callable
import logging
import numpy as np
from .population import Population, PopulationQueryItem, desc_similarity, score_to_probability_with_logarithmic_curve
from .es_population import ESPopulation
from ..individual import Individual


class EnsemblePopulation(Population):
    """
    An ensemble of populations. It is a population itself, but it contains other populations.
    """

    def __init__(self, populations: list[Population], weights: list[float], query_item: PopulationQueryItem):
        """
        Constructor.

        :param populations: The populations that are part of the ensemble.
        :param weights: The weights of the populations. They should sum 1.
        :param query_item: The query item of the ensemble.
        """
        super().__init__(query_item)
        self.populations = populations
        self.weights = weights
        self.logger = logging.getLogger(__name__)

    def __len__(self) -> int:
        """
        Returns the number of individuals in the ensemble.

        :return: The number of individuals in the ensemble.
        """
        return sum(len(population) for population in self.populations)

    def __getitem__(self, index: int) -> Individual:
        """
        Returns the individual at the given index in the ensemble.

        :param index: The index of the individual.
        :return: The individual at the given index in the ensemble.
        """
        for population in self.populations:
            if index < len(population):
                return population[index]
            index -= len(population)
        raise IndexError("Index out of bounds")

    def __iter__(self):
        """
        Returns an iterator over the individuals in the ensemble.

        :return: An iterator over the individuals in the ensemble.
        """
        for population in self.populations:
            for individual in population:
                yield individual

    def __contains__(self, individual: Individual) -> bool:
        """
        Returns whether the given individual is in the ensemble.

        :param individual: The individual to check.
        :return: Whether the given individual is in the ensemble.
        """
        for population in self.populations:
            if individual in population:
                return True
        return False

    def append(self, individual: Individual):
        """
        Appends the given individual to the ensemble.

        :param individual: The individual to append.
        """
        raise NotImplementedError("Cannot append to ensemble population")

    def extend(self, individuals: list[Individual]):
        """
        Extends the ensemble with the given individuals.

        :param individuals: The individuals to extend the ensemble with.
        """
        raise NotImplementedError("Cannot extend ensemble population")

    def remove(self, individual: Individual):
        """
        Removes the given individual from the ensemble.
        