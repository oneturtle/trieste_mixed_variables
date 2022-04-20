# Copyright 2020 The Trieste Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" This module contains implementations of various types of search space. """
from __future__ import annotations
from lib2to3.pytree import convert

import operator
from abc import ABC, abstractmethod
from functools import reduce
from typing import Optional, Sequence, TypeVar, overload
from markupsafe import string
from numpy import dtype, zeros
import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

from .types import TensorType
from .utils import shapes_equal

SearchSpaceType = TypeVar("SearchSpaceType", bound="SearchSpace")
""" A type variable bound to :class:`SearchSpace`. """


class SearchSpace(ABC):
    """
    A :class:`SearchSpace` represents the domain over which an objective function is optimized.
    """

    @abstractmethod
    def sample(self, num_samples: int) -> TensorType:
        """
        :param num_samples: The number of points to sample from this search space.
        :return: ``num_samples`` i.i.d. random points, sampled uniformly from this search space.
        """

    @abstractmethod
    def __contains__(self, value: TensorType) -> bool | TensorType:
        """
        :param value: A point to check for membership of this :class:`SearchSpace`.
        :return: `True` if ``value`` is a member of this search space, else `False`. May return a
            scalar boolean `TensorType` instead of the `bool` itself.
        :raise ValueError (or tf.errors.InvalidArgumentError): If ``value`` has a different
            dimensionality from this :class:`SearchSpace`.
        """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """The number of inputs in this search space."""

    @property
    @abstractmethod
    def lower(self) -> TensorType:
        """The lowest value taken by each search space dimension."""

    @property
    @abstractmethod
    def upper(self) -> TensorType:
        """The highest value taken by each search space dimension."""

    @abstractmethod
    def __mul__(self: SearchSpaceType, other: SearchSpaceType) -> SearchSpaceType:
        """
        :param other: A search space of the same type as this search space.
        :return: The Cartesian product of this search space with the ``other``.
        """

    def __pow__(self: SearchSpaceType, other: int) -> SearchSpaceType:
        """
        Return the Cartesian product of ``other`` instances of this search space. For example, for
        an exponent of `3`, and search space `s`, this is `s ** 3`, which is equivalent to
        `s * s * s`.

        :param other: The exponent, or number of instances of this search space to multiply
            together. Must be strictly positive.
        :return: The Cartesian product of ``other`` instances of this search space.
        :raise tf.errors.InvalidArgumentError: If the exponent ``other`` is less than 1.
        """
        tf.debugging.assert_positive(other, message="Exponent must be strictly positive")
        return reduce(operator.mul, [self] * other)


class DiscreteSearchSpace(SearchSpace):
    r"""
    A discrete :class:`SearchSpace` representing a finite set of :math:`D`-dimensional points in
    :math:`\mathbb{R}^D`.

    For example:

        >>> points = tf.constant([[-1.0, 0.4], [-1.0, 0.6], [0.0, 0.4]])
        >>> search_space = DiscreteSearchSpace(points)
        >>> assert tf.constant([0.0, 0.4]) in search_space
        >>> assert tf.constant([1.0, 0.5]) not in search_space

    """

    def __init__(self, points: TensorType):
        """
        :param points: The points that define the discrete space, with shape ('N', 'D').
        :raise ValueError (or tf.errors.InvalidArgumentError): If ``points`` has an invalid shape.
        """

        tf.debugging.assert_shapes([(points, ("N", "D"))])
        self._points = points
        self._dimension = tf.shape(self._points)[-1]

    def __repr__(self) -> str:
        """"""
        return f"DiscreteSearchSpace({self._points!r})"

    @property
    def lower(self) -> TensorType:
        """The lowest value taken across all points by each search space dimension."""
        return tf.reduce_min(self.points, -2)

    @property
    def upper(self) -> TensorType:
        """The highest value taken across all points by each search space dimension."""
        return tf.reduce_max(self.points, -2)

    @property
    def points(self) -> TensorType:
        """All the points in this space."""
        return self._points

    @property
    def dimension(self) -> int:
        """The number of inputs in this search space."""
        return self._dimension

    def __contains__(self, value: TensorType) -> bool | TensorType:
        tf.debugging.assert_shapes([(value, self.points.shape[1:])])
        return tf.reduce_any(tf.reduce_all(value == self._points, axis=1))

    def sample(self, num_samples: int) -> TensorType:
        """
        :param num_samples: The number of points to sample from this search space.
        :return: ``num_samples`` i.i.d. random points, sampled uniformly,
            from this search space.
        """
        if num_samples == 0:
            return self.points[:0, :]
        else:
            sampled_indices = tf.random.categorical(
                tf.ones((1, tf.shape(self.points)[0])), num_samples
            )
            return tf.gather(self.points, sampled_indices)[0, :, :]  # [num_samples, D]

    def __mul__(self, other: DiscreteSearchSpace) -> DiscreteSearchSpace:
        r"""
        Return the Cartesian product of the two :class:`DiscreteSearchSpace`\ s. For example:

            >>> sa = DiscreteSearchSpace(tf.constant([[0, 1], [2, 3]]))
            >>> sb = DiscreteSearchSpace(tf.constant([[4, 5, 6], [7, 8, 9]]))
            >>> (sa * sb).points.numpy()
            array([[0, 1, 4, 5, 6],
                   [0, 1, 7, 8, 9],
                   [2, 3, 4, 5, 6],
                   [2, 3, 7, 8, 9]], dtype=int32)

        :param other: A :class:`DiscreteSearchSpace` with :attr:`points` of the same dtype as this
            search space.
        :return: The Cartesian product of the two :class:`DiscreteSearchSpace`\ s.
        :raise TypeError: If one :class:`DiscreteSearchSpace` has :attr:`points` of a different
            dtype to the other.
        """
        if self.points.dtype is not other.points.dtype:
            return NotImplemented

        tile_self = tf.tile(self.points[:, None], [1, len(other.points), 1])
        tile_other = tf.tile(other.points[None], [len(self.points), 1, 1])
        cartesian_product = tf.concat([tile_self, tile_other], axis=2)
        product_space_dimension = self.points.shape[-1] + other.points.shape[-1]
        return DiscreteSearchSpace(tf.reshape(cartesian_product, [-1, product_space_dimension]))

    def __deepcopy__(self, memo: dict[int, object]) -> DiscreteSearchSpace:
        return self


class Box(SearchSpace):
    r"""
    Continuous :class:`SearchSpace` representing a :math:`D`-dimensional box in
    :math:`\mathbb{R}^D`. Mathematically it is equivalent to the Cartesian product of :math:`D`
    closed bounded intervals in :math:`\mathbb{R}`.
    """

    @overload
    def __init__(self, lower: Sequence[float], upper: Sequence[float]):
        ...

    @overload
    def __init__(self, lower: TensorType, upper: TensorType):
        ...

    def __init__(
        self,
        lower: Sequence[float] | TensorType,
        upper: Sequence[float] | TensorType,
    ):
        r"""
        If ``lower`` and ``upper`` are `Sequence`\ s of floats (such as lists or tuples),
        they will be converted to tensors of dtype `tf.float64`.

        :param lower: The lower (inclusive) bounds of the box. Must have shape [D] for positive D,
            and if a tensor, must have float type.
        :param upper: The upper (inclusive) bounds of the box. Must have shape [D] for positive D,
            and if a tensor, must have float type.
        :param varTypes: The types of variables, such as "Continue" or "Discrete".
        :raise ValueError (or tf.errors.InvalidArgumentError): If any of the following are true:

            - ``lower`` and ``upper`` have invalid shapes.
            - ``lower`` and ``upper`` do not have the same floating point type.
            - ``upper`` is not greater than ``lower`` across all dimensions.
        """

        tf.debugging.assert_shapes([(lower, ["D"]), (upper, ["D"])])
        tf.assert_rank(lower, 1)
        tf.assert_rank(upper, 1)

        if isinstance(lower, Sequence):
            self._lower = tf.constant(lower, dtype=tf.float64)
            self._upper = tf.constant(upper, dtype=tf.float64)
        else:
            self._lower = tf.convert_to_tensor(lower)
            self._upper = tf.convert_to_tensor(upper)

            tf.debugging.assert_same_float_dtype([self._lower, self._upper])

        tf.debugging.assert_less(self._lower, self._upper)

        self._dimension = tf.shape(self._upper)[-1]

    def __repr__(self) -> str:
        """"""
        return f"Box({self._lower!r}, {self._upper!r})"

    @property
    def lower(self) -> TensorType:
        """The lower bounds of the box."""
        return self._lower

    @property
    def upper(self) -> TensorType:
        """The upper bounds of the box."""
        return self._upper

    @property
    def dimension(self) -> int:
        """The number of inputs in this search space."""
        return self._dimension

    def __contains__(self, value: TensorType) -> bool | TensorType:
        """
        Return `True` if ``value`` is a member of this search space, else `False`. A point is a
        member if all of its coordinates lie in the closed intervals bounded by the lower and upper
        bounds.

        :param value: A point to check for membership of this :class:`SearchSpace`.
        :return: `True` if ``value`` is a member of this search space, else `False`. May return a
            scalar boolean `TensorType` instead of the `bool` itself.
        :raise ValueError (or tf.errors.InvalidArgumentError): If ``value`` has a different
            dimensionality from the search space.
        """

        tf.debugging.assert_equal(
            shapes_equal(value, self._lower),
            True,
            message=f"""
                Dimensionality mismatch: space is {self._lower}, value is {tf.shape(value)}
                """,
        )

        return tf.reduce_all(value >= self._lower) and tf.reduce_all(value <= self._upper)

    def sample(self, num_samples: int) -> TensorType:
        """
        Sample randomly from the space.

        :param num_samples: The number of points to sample from this search space.
        :return: ``num_samples`` i.i.d. random points, sampled uniformly,
            from this search space with shape '[num_samples, D]' , where D is the search space
            dimension.
        """
        tf.debugging.assert_non_negative(num_samples)

        dim = tf.shape(self._lower)[-1]
        return tf.random.uniform(
            (num_samples, dim), minval=self._lower, maxval=self._upper, dtype=self._lower.dtype
        )

    def sample_halton(self, num_samples: int, seed: Optional[int] = None) -> TensorType:
        """
        Sample from the space using a Halton sequence. The resulting samples are guaranteed to be
        diverse and are reproducible by using the same choice of ``seed``.

        :param num_samples: The number of points to sample from this search space.
        :param seed: Random seed for the halton sequence
        :return: ``num_samples`` of points, using halton sequence with shape '[num_samples, D]' ,
            where D is the search space dimension.
        """

        tf.debugging.assert_non_negative(num_samples)
        if num_samples == 0:
            return tf.constant([])
        if seed is not None:  # ensure reproducibility
            tf.random.set_seed(seed)
        dim = tf.shape(self._lower)[-1]
        return (self._upper - self._lower) * tfp.mcmc.sample_halton_sequence(
            dim=dim, num_results=num_samples, dtype=self._lower.dtype, seed=seed
        ) + self._lower

    def sample_sobol(self, num_samples: int, skip: Optional[int] = None) -> TensorType:
        """
        Sample a diverse set from the space using a Sobol sequence.
        If ``skip`` is specified, then the resulting samples are reproducible.

        :param num_samples: The number of points to sample from this search space.
        :param skip: The number of initial points of the Sobol sequence to skip
        :return: ``num_samples`` of points, using sobol sequence with shape '[num_samples, D]' ,
            where D is the search space dimension.
        """
        tf.debugging.assert_non_negative(num_samples)
        if num_samples == 0:
            return tf.constant([])
        if skip is None:  # generate random skip
            skip = tf.random.uniform([1], maxval=2 ** 16, dtype=tf.int32)[0]
        dim = tf.shape(self._lower)[-1]
        return (self._upper - self._lower) * tf.math.sobol_sample(
            dim=dim, num_results=num_samples, dtype=self._lower.dtype, skip=skip
        ) + self._lower

    def discretize(self, num_samples: int) -> DiscreteSearchSpace:
        """
        :param num_samples: The number of points in the :class:`DiscreteSearchSpace`.
        :return: A discrete search space consisting of ``num_samples`` points sampled uniformly from
            this :class:`Box`.
        """
        return DiscreteSearchSpace(points=self.sample(num_samples))

    def __mul__(self, other: Box) -> Box:
        r"""
        Return the Cartesian product of the two :class:`Box`\ es (concatenating their respective
        lower and upper bounds). For example:

            >>> unit_interval = Box([0.0], [1.0])
            >>> square_at_origin = Box([-2.0, -2.0], [2.0, 2.0])
            >>> new_box = unit_interval * square_at_origin
            >>> new_box.lower.numpy()
            array([ 0., -2., -2.])
            >>> new_box.upper.numpy()
            array([1., 2., 2.])

        :param other: A :class:`Box` with bounds of the same type as this :class:`Box`.
        :return: The Cartesian product of the two :class:`Box`\ es.
        :raise TypeError: If the bounds of one :class:`Box` have different dtypes to those of
            the other :class:`Box`.
        """
        if self.lower.dtype is not other.lower.dtype:
            return NotImplemented

        product_lower_bound = tf.concat([self._lower, other.lower], axis=-1)
        product_upper_bound = tf.concat([self._upper, other.upper], axis=-1)

        return Box(product_lower_bound, product_upper_bound)

    def __deepcopy__(self, memo: dict[int, object]) -> Box:
        return self


class TaggedProductSearchSpace(SearchSpace):
    r"""
    Product :class:`SearchSpace` consisting of a product of
    multiple :class:`SearchSpace`. This class provides functionality for
    accessing either the resulting combined search space or each individual space.

    Note that this class assumes that individual points in product spaces are
    represented with their inputs in the same order as specified when initializing
    the space.
    """

    def __init__(self, spaces: Sequence[SearchSpace], tags: Optional[Sequence[str]] = None):
        r"""
        Build a :class:`TaggedProductSearchSpace` from a list ``spaces`` of other spaces. If
        ``tags`` are provided then they form the identifiers of the subspaces, otherwise the
        subspaces are labelled numerically.

        :param spaces: A sequence of :class:`SearchSpace` objects representing the space's subspaces
        :param tags: An optional list of tags giving the unique identifiers of
            the space's subspaces.
        :raise ValueError (or tf.errors.InvalidArgumentError): If ``spaces`` has a different
            length to ``tags`` when ``tags`` is provided or if ``tags`` contains duplicates.
        """

        number_of_subspaces = len(spaces)
        if tags is None:
            tags = [str(index) for index in range(number_of_subspaces)]
        else:
            number_of_tags = len(tags)
            tf.debugging.assert_equal(
                number_of_tags,
                number_of_subspaces,
                message=f"""
                    Number of tags must match number of subspaces but
                    received {number_of_tags} tags and {number_of_subspaces} subspaces.
                """,
            )
            number_of_unique_tags = len(set(tags))
            tf.debugging.assert_equal(
                number_of_tags,
                number_of_unique_tags,
                message=f"Subspace names must be unique but received {tags}.",
            )

        self._spaces = dict(zip(tags, spaces))

        subspace_sizes = [space.dimension for space in spaces]

        self._subspace_sizes_by_tag = {
            tag: subspace_size for tag, subspace_size in zip(tags, subspace_sizes)
        }

        self._subspace_starting_indices = dict(zip(tags, tf.cumsum(subspace_sizes, exclusive=True)))

        self._dimension = tf.reduce_sum(subspace_sizes)
        self._tags = tuple(tags)  # avoid accidental modification by users

    def __repr__(self) -> str:
        """"""
        return f"""TaggedProductSearchSpace(spaces =
                {[self.get_subspace(tag) for tag in self.subspace_tags]},
                tags = {self.subspace_tags})
                """

    @property
    def lower(self) -> TensorType:
        """The lowest values taken by each space dimension, concatenated across subspaces."""
        lower_for_each_subspace = [self.get_subspace(tag).lower for tag in self.subspace_tags]
        return tf.concat(lower_for_each_subspace, axis=-1)

    @property
    def upper(self) -> TensorType:
        """The highest values taken by each space dimension, concatenated across subspaces."""
        upper_for_each_subspace = [self.get_subspace(tag).upper for tag in self.subspace_tags]
        return tf.concat(upper_for_each_subspace, axis=-1)

    @property
    def subspace_tags(self) -> tuple[str, ...]:
        """Return the names of the subspaces contained in this product space."""
        return self._tags

    @property
    def dimension(self) -> int:
        """The number of inputs in this product search space."""
        return int(self._dimension)

    def get_subspace(self, tag: str) -> SearchSpace:
        """
        Return the domain of a particular subspace.

        :param tag: The tag specifying the target subspace.
        """
        tf.debugging.assert_equal(
            tag in self.subspace_tags,
            True,
            message=f"""
                Attempted to access a subspace that does not exist. This space only contains
                subspaces with the tags {self.subspace_tags} but received {tag}.
            """,
        )
        return self._spaces[tag]

    def fix_subspace(self, tag: str, values: TensorType) -> TaggedProductSearchSpace:
        """
        Return a new :class:`TaggedProductSearchSpace` with the specified subspace replaced with
        a :class:`DiscreteSearchSpace` containing ``values`` as its points. This is useful if you
        wish to restrict subspaces to sets of representative points.

        :param tag: The tag specifying the target subspace.
        :param values: The  values used to populate the new discrete subspace.z
        """

        new_spaces = [
            self.get_subspace(t) if t != tag else DiscreteSearchSpace(points=values)
            for t in self.subspace_tags
        ]

        return TaggedProductSearchSpace(spaces=new_spaces, tags=self.subspace_tags)

    def get_subspace_component(self, tag: str, values: TensorType) -> TensorType:
        """
        Returns the components of ``values`` lying in a particular subspace.

        :param value: Points from the :class:`TaggedProductSearchSpace` of shape [N,Dprod].
        :return: The sub-components of ``values`` lying in the specified subspace, of shape
            [N, Dsub], where Dsub is the dimensionality of the specified subspace.
        """

        starting_index_of_subspace = self._subspace_starting_indices[tag]
        ending_index_of_subspace = starting_index_of_subspace + self._subspace_sizes_by_tag[tag]
        return values[:, starting_index_of_subspace:ending_index_of_subspace]

    def __contains__(self, value: TensorType) -> bool | TensorType:
        """
        Return `True` if ``value`` is a member of this search space, else `False`. A point is a
        member if each of its subspace components lie in each subspace.

        Recall that individual points in product spaces are represented with their inputs in the
        same order as specified when initializing the space.

        :param value: A point to check for membership of this :class:`SearchSpace`.
        :return: `True` if ``value`` is a member of this search space, else `False`. May return a
            scalar boolean `TensorType` instead of the `bool` itself.
        :raise ValueError (or tf.errors.InvalidArgumentError): If ``value`` has a different
            dimensionality from the search space.
        """

        tf.debugging.assert_equal(
            tf.shape(value),
            self.dimension,
            message=f"""
                Dimensionality mismatch: space is {self.dimension}, value is {tf.shape(value)}
                """,
        )
        value = value[tf.newaxis, ...]
        in_each_subspace = [
            self._spaces[tag].__contains__(self.get_subspace_component(tag, value)[0, :])
            for tag in self._tags
        ]
        return tf.reduce_all(in_each_subspace)

    def sample(self, num_samples: int) -> TensorType:
        """
        Sample randomly from the space by sampling from each subspace
        and concatenating the resulting samples.

        :param num_samples: The number of points to sample from this search space.
        :return: ``num_samples`` i.i.d. random points, sampled uniformly,
            from this search space with shape '[num_samples, D]' , where D is the search space
            dimension.
        """
        tf.debugging.assert_non_negative(num_samples)

        subspace_samples = [self._spaces[tag].sample(num_samples) for tag in self._tags]
        return tf.concat(subspace_samples, -1)

    def discretize(self, num_samples: int) -> DiscreteSearchSpace:
        """
        :param num_samples: The number of points in the :class:`DiscreteSearchSpace`.
        :return: A discrete search space consisting of ``num_samples`` points sampled
            uniformly across the space.
        """
        return DiscreteSearchSpace(points=self.sample(num_samples))

    def __mul__(self, other: SearchSpace) -> TaggedProductSearchSpace:
        r"""
        Return the Cartesian product of the two :class:`TaggedProductSearchSpace`\ s,
        building a tree of :class:`TaggedProductSearchSpace`\ s.

        :param other: A search space of the same type as this search space.
        :return: The Cartesian product of this search space with the ``other``.
        """
        return TaggedProductSearchSpace(spaces=[self, other])

    def __deepcopy__(self, memo: dict[int, object]) -> TaggedProductSearchSpace:
        return self


class Box_mixed(SearchSpace):
    r"""
    Continuous or Discretes :class:`SearchSpace` representing a :math:`D`-dimensional box in
    :math:`\mathbb{R}^D`. Mathematically it is equivalent to the Cartesian product of :math:`D`
    closed bounded intervals in :math:`\mathbb{R}`.
    """

    @overload
    def __init__(self, lower: Sequence[float], upper: Sequence[float], varTypes: Sequence[str], known_constraints=None):
        ...

    @overload
    def __init__(self, lower: TensorType, upper: TensorType, varTypes: TensorType, known_constraints=None):
        ...

    def __init__(
        self,
        lower: Sequence[float] | TensorType,
        upper: Sequence[float] | TensorType,
        varTypes: Sequence[str] | TensorType,
        known_constraints = None,
    ):
        r"""
        If ``lower`` and ``upper`` are `Sequence`\ s of floats (such as lists or tuples),
        they will be converted to tensors of dtype `tf.float64`.

        :param lower: The lower (inclusive) bounds of the box. Must have shape [D] for positive D,
            and if a tensor, must have float type.
        :param upper: The upper (inclusive) bounds of the box. Must have shape [D] for positive D,
            and if a tensor, must have float type.
        :param varTypes: The types of variables, such as "Continuous" or "Discrete".
        :param known_constraints: The constraints which can be listed explicity. A list of dict,
            such as known_constraints = [{'type': 'ineq', 'fun': '-x[:,1] -.5 + abs(x[:,0]) - np.sqrt(1-x[:,0]**2)'},
                                        {'type': 'eq', 'fun': 'x[:,1] +.5 - abs(x[:,0]) - np.sqrt(1-x[:,0]**2)'}]
        :raise ValueError (or tf.errors.InvalidArgumentError): If any of the following are true:

            - ``lower`` and ``upper`` have invalid shapes.
            - ``lower`` and ``upper`` do not have the same floating point type.
            - ``upper`` is not greater than ``lower`` across all dimensions.
        """
        tf.debugging.assert_shapes([(lower, ["D"]), (upper, ["D"]), (varTypes, ["D"])])
        tf.assert_rank(lower, 1)
        tf.assert_rank(upper, 1)
        tf.assert_rank(varTypes, 1)
        
        
        if isinstance(lower, Sequence):
            self._lower = tf.constant(lower, dtype=tf.float64)
            self._upper = tf.constant(upper, dtype=tf.float64)
            self._varTypes = tf.constant(varTypes, dtype=tf.string)
        else:
            self._lower = tf.convert_to_tensor(lower)
            self._upper = tf.convert_to_tensor(upper)
            self._varTypes = tf.convert_to_tensor(varTypes)

            tf.debugging.assert_same_float_dtype([self._lower, self._upper])

        tf.debugging.assert_less(self._lower, self._upper)

        self._dimension = tf.shape(self._upper)[-1]

        self.known_constraints = known_constraints

    def __repr__(self) -> str:
        """"""
        return f"Box({self._lower!r}, {self._upper!r}, {self._varTypes!r})"

    @property
    def lower(self) -> TensorType:
        """The lower bounds of the box."""
        return self._lower

    @property
    def upper(self) -> TensorType:
        """The upper bounds of the box."""
        return self._upper
    
    @property
    def varTypes(self) -> TensorType:
        """The types of variables."""
        return self._varTypes

    @property
    def dimension(self) -> int:
        """The number of inputs in this search space."""
        return self._dimension

    def __contains__(self, value: TensorType) -> bool | TensorType:
        """
        Return `True` if ``value`` is a member of this search space, else `False`. A point is a
        member if all of its coordinates lie in the closed intervals bounded by the lower and upper
        bounds.

        :param value: A point to check for membership of this :class:`SearchSpace`.
        :return: `True` if ``value`` is a member of this search space, else `False`. May return a
            scalar boolean `TensorType` instead of the `bool` itself.
        :raise ValueError (or tf.errors.InvalidArgumentError): If ``value`` has a different
            dimensionality from the search space.
        """

        tf.debugging.assert_equal(
            shapes_equal(value, self._lower),
            True,
            message=f"""
                Dimensionality mismatch: space is {self._lower}, value is {tf.shape(value)}
                """,
        )

        return tf.reduce_all(value >= self._lower) and tf.reduce_all(value <= self._upper)

    def sample(self, num_samples: int) -> TensorType:
        """
        Sample randomly from the space with known_constraints.

        :param num_samples: The number of points to sample from this search space.
        :return: ``num_samples`` i.i.d. random points, sampled uniformly,
            from this search space with shape '[num_samples, D]' , where D is the search space
            dimension.
        """
        tf.debugging.assert_non_negative(num_samples)

        dim = tf.shape(self._lower)[-1]
        samples = np.empty((num_samples, dim))
        
        if self.has_known_constraints():
            samples = self.get_samples_with_constraints(num_samples)
        else:
            samples = self.get_samples_without_constraints(num_samples)
        
        #return samples
        return tf.Variable(samples)
    
    def smaple_latin(self, num_samples: int, criterion: str = 'center'):
        """
        Latin experiment design for initial experiment design.
        Uses random design for non-continuous variables, and latin hypercube for continuous ones
        
        :param num_samples: Number of samples to generate
        :param criterion: For details of the effect of this parameter, please refer to pyDOE.lhs documentation
                          Default: 'center'
        :returns: Generated samples
        """
        tf.debugging.assert_non_negative(num_samples)

        dim = tf.shape(self._lower)[-1]
        samples = np.empty((num_samples, dim))

        for (idx, var) in enumerate(self._varTypes):
            if var=="Discrete" or var=="discrete" :
                sample_var = np.atleast_2d(np.random.choice(list(np.arange(self._lower[idx], self._upper[idx]+1)), num_samples))
                samples[:,idx] = sample_var.flatten()
        
        if self.has_continuous():
            bounds = self.get_continuous_bounds()
            lower_bound = np.asarray(bounds)[:,0].reshape(1, len(bounds))
            upper_bound = np.asarray(bounds)[:,1].reshape(1, len(bounds))
            diff = upper_bound - lower_bound

            from pyDOE import lhs
            X_design_aux = lhs(len(bounds), num_samples, criterion=criterion)
            I = np.ones((X_design_aux.shape[0], 1))
            X_design = np.dot(I, lower_bound) + X_design_aux * np.dot(I, diff)

            samples[:, self.get_continuous_dims()] = X_design
        
        #return samples
        return tf.Variable(samples)

    def sample_halton(self, num_samples: int, seed: Optional[int] = None) -> TensorType:
        """
        Sample from the space using a Halton sequence. The resulting samples are guaranteed to be
        diverse and are reproducible by using the same choice of ``seed``.

        :param num_samples: The number of points to sample from this search space.
        :param seed: Random seed for the halton sequence
        :return: ``num_samples`` of points, using halton sequence with shape '[num_samples, D]' ,
            where D is the search space dimension.
        """

        tf.debugging.assert_non_negative(num_samples)
        if num_samples == 0:
            return tf.constant([])
        if seed is not None:  # ensure reproducibility
            tf.random.set_seed(seed)
        dim = tf.shape(self._lower)[-1]
        return (self._upper - self._lower) * tfp.mcmc.sample_halton_sequence(
            dim=dim, num_results=num_samples, dtype=self._lower.dtype, seed=seed
        ) + self._lower

    def sample_sobol(self, num_samples: int, skip: Optional[int] = None) -> TensorType:
        """
        Sample a diverse set from the space using a Sobol sequence.
        If ``skip`` is specified, then the resulting samples are reproducible.

        :param num_samples: The number of points to sample from this search space.
        :param skip: The number of initial points of the Sobol sequence to skip
        :return: ``num_samples`` of points, using sobol sequence with shape '[num_samples, D]' ,
            where D is the search space dimension.
        """
        tf.debugging.assert_non_negative(num_samples)
        if num_samples == 0:
            return tf.constant([])
        if skip is None:  # generate random skip
            skip = tf.random.uniform([1], maxval=2 ** 16, dtype=tf.int32)[0]
        dim = tf.shape(self._lower)[-1]
        return (self._upper - self._lower) * tf.math.sobol_sample(
            dim=dim, num_results=num_samples, dtype=self._lower.dtype, skip=skip
        ) + self._lower

    def discretize(self, num_samples: int) -> DiscreteSearchSpace:
        """
        :param num_samples: The number of points in the :class:`DiscreteSearchSpace`.
        :return: A discrete search space consisting of ``num_samples`` points sampled uniformly from
            this :class:`Box`.
        """
        return DiscreteSearchSpace(points=self.sample(num_samples))

    def __mul__(self, other: Box) -> Box:
        r"""
        Return the Cartesian product of the two :class:`Box`\ es (concatenating their respective
        lower and upper bounds). For example:

            >>> unit_interval = Box([0.0], [1.0])
            >>> square_at_origin = Box([-2.0, -2.0], [2.0, 2.0])
            >>> new_box = unit_interval * square_at_origin
            >>> new_box.lower.numpy()
            array([ 0., -2., -2.])
            >>> new_box.upper.numpy()
            array([1., 2., 2.])

        :param other: A :class:`Box` with bounds of the same type as this :class:`Box`.
        :return: The Cartesian product of the two :class:`Box`\ es.
        :raise TypeError: If the bounds of one :class:`Box` have different dtypes to those of
            the other :class:`Box`.
        """
        if self.lower.dtype is not other.lower.dtype:
            return NotImplemented

        product_lower_bound = tf.concat([self._lower, other.lower], axis=-1)
        product_upper_bound = tf.concat([self._upper, other.upper], axis=-1)

        return Box(product_lower_bound, product_upper_bound)

    def __deepcopy__(self, memo: dict[int, object]) -> Box:
        return self

    def get_samples_with_constraints(self, num_samples):
        """
        Draw random samples and only save those that satisfy constraints
        Finish when required number of samples is generated
        """
        samples = np.empty((0, self._dimension))

        while samples.shape[0] < num_samples:
            domain_samples = self.get_samples_without_constraints(num_samples)
            valid_indices = (self.indicator_constraints(domain_samples) == 1).flatten()
            if sum(valid_indices) > 0:
                valid_samples = domain_samples[valid_indices,:]
                samples = np.vstack((samples,valid_samples))

        return samples[0:num_samples,:]
    
    def fill_noncontinous_variables(self, samples):
        """
        Fill sample values to non-continuous variables in place
        """
        num_samples = samples.shape[0]
        for (idx, var) in enumerate(self._varTypes):
            if var=="Discrete" or var=="discrete" :
                sample_var = np.atleast_2d(np.random.choice(list(np.arange(self._lower[idx], self._upper[idx]+1)), num_samples))
                samples[:,idx] = sample_var.flatten()
    
    def get_samples_without_constraints(self, num_samples):
        samples = np.empty((num_samples, self._dimension))

        self.fill_noncontinous_variables(samples)

        if self.has_continuous():
            X_design = self.samples_multidimensional_uniform(self.get_continuous_bounds(), num_samples)
            samples[:, self.get_continuous_dims()] = X_design

        return samples
    
    def samples_multidimensional_uniform(self, bounds, points_count):
        """
        Generates a multidimensional grid uniformly distributed.
        :param bounds: tuple defining the box constraints.
        :points_count: number of data points to generate.
        """
        dim = len(bounds)
        Z_rand = np.zeros(shape=(points_count, dim))
        for k in range(0,dim):
            Z_rand[:,k] = np.random.uniform(low=bounds[k][0], high=bounds[k][1], size=points_count)
        return Z_rand

    def indicator_constraints(self,x):
        """
        Returns array of ones and zeros indicating if x is within the constraints
        """
        x = np.atleast_2d(x)    #[num_samples, D]
        I_x = np.ones((x.shape[0],1))
        if self.known_constraints is not None:
            for d in self.known_constraints:
                try:
                    #exec('known_constraint = lambda x:' + d['known_constraint'], globals())
                    #ind_x = (known_constraint(x) <= 0).numpy() * 1
                    ind_x = d.evaluate(x) * 1
                    I_x *= ind_x.reshape(x.shape[0],1)
                except:
                    print('Fail to compile the constraint: ' + str(d))
                    raise
        return I_x

    def has_known_constraints(self):
        """
        Checks if the problem has known_constraints. Note that the coordinates of the constraints are defined
        in terms of the model inputs and not in terms of the objective inputs. This means that if bandit or
        discre varaibles are in place, the restrictions should reflect this fact (TODO: implement the
        mapping of constraints defined on the objective to constraints defined on the model).
        """
        return self.known_constraints is not None

    def has_continuous(self):
        """
        Returns `true` if the space contains at least one continuous variable, and `false` otherwise
        """
        return any(v =="continuous" or v == "Continuous" for v in self._varTypes)
    
    def get_continuous_bounds(self):
        """
        Extracts the bounds of the continuous variables.
        """
        bounds = []
        for idx, d in enumerate(self._varTypes):
            if d == 'continuous' or d == "Continuous":
                bounds.append([self._lower[idx], self._upper[idx]])
        return bounds

    def get_continuous_dims(self):
        """
        Returns the dimension of the continuous components of the space.
        """
        continuous_dims = []
        for i in range(tf.shape(self._lower)[-1]):
            if self._varTypes[i] == 'continuous' or self._varTypes[i] == "Continuous":
                continuous_dims += [i]
        return continuous_dims
    
    def round_optimum(self, x: TensorType) -> TensorType:
        """
        Rounds some value x to a feasible value in the design space.
        x is expected to be a vector or an array with a single row
        param: x is tensorflow with [num_optimization_runs, V, D]
        """
        
        if not (x.ndim == 3):
            raise ValueError("Unexpected dimentionality of x. Got {}, expected (num_optimization_runs, V, D)".format(x.ndim))
        
        rounded_design_x = np.zeros(x.shape) # [num_optimization_runs, V, D]
        for i in range(x.shape[1]):
            design_x = x[:,i,:]  # [num_optimization_runs, D]
            for j in range(x.shape[0]):
                one_x = design_x[j, :]  # [1, D]

            one_x = np.array(one_x)
            if not ((one_x.ndim == 1) or (one_x.ndim == 2 and one_x.shape[0] == 1)):
                raise ValueError("Unexpected dimentionality of x. Got {}, expected (1, N) or (N,)".format(one_x.ndim))

            if one_x.ndim == 2:
                one_x = one_x[0]

            for idx, varType in enumerate(self._varTypes):
                var_value = one_x[idx]
                lower = self._lower[idx]
                upper = self._upper[idx]
                if varType == "continuous" or varType == "Continuous":
                    var_value_rounded = self.round_continuous(lower, upper, var_value)
                if varType == "discrete" or varType == "Discrete":
                    var_value_rounded = self.round_discrete(lower, upper, var_value)
                
                rounded_design_x[j, i, idx] = var_value_rounded

        return tf.constant(rounded_design_x)
    
    def round_continuous(self, lower, upper, value_array):
        """
        If value falls within bounds, just return it
        otherwise return min or max, whichever is closer to the value
        Assumes an 1d array with a single element as an input.
        """

        min_value = lower
        max_value = upper

        rounded_value = value_array
        if rounded_value < min_value:
            rounded_value = min_value
        elif rounded_value > max_value:
            rounded_value = max_value

        return rounded_value
    
    def round_discrete(self, lower, upper, value_array):
        """
        Rounds a discrete variable by selecting the closest point in the domain
        Assumes an 1d array with a single element as an input.
        """
        value = value_array
        domain = np.arange(lower, upper+1)
        rounded_value = domain[0]

        for domain_value in domain:
            if np.abs(domain_value - value) < np.abs(rounded_value - value):
                rounded_value = domain_value

        return rounded_value