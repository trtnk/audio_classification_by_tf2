# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Image warping using sparse flow defined at control points."""

import numpy as np
import tensorflow as tf
from tensorflow_addons.image import dense_image_warp
from tensorflow_addons.utils.types import TensorLike, FloatTensorLike

EPSILON = 0.0000000001


def _get_grid_locations(
        image_height: TensorLike, image_width: TensorLike
) -> TensorLike:
    """Wrapper for `np.meshgrid`."""

    y_range = np.linspace(0, image_height - 1, image_height)
    x_range = np.linspace(0, image_width - 1, image_width)
    y_grid, x_grid = np.meshgrid(y_range, x_range, indexing="ij")
    return np.stack((y_grid, x_grid), -1)


def _expand_to_minibatch(np_array: TensorLike, batch_size: TensorLike) -> TensorLike:
    """Tile arbitrarily-sized np_array to include new batch dimension."""
    tiles = [batch_size] + [1] * np_array.ndim
    return np.tile(np.expand_dims(np_array, 0), tiles)


def _get_boundary_locations(
        image_height: TensorLike, image_width: TensorLike, num_points_per_edge: TensorLike,
) -> TensorLike:
    """Compute evenly-spaced indices along edge of image."""
    y_range = np.linspace(0, image_height - 1, num_points_per_edge + 2)
    x_range = np.linspace(0, image_width - 1, num_points_per_edge + 2)
    ys, xs = np.meshgrid(y_range, x_range, indexing="ij")
    is_boundary = np.logical_or(
        np.logical_or(xs == 0, xs == image_width - 1),
        np.logical_or(ys == 0, ys == image_height - 1),
    )
    return np.stack([ys[is_boundary], xs[is_boundary]], axis=-1)


def _add_zero_flow_controls_at_boundary(
        control_point_locations: TensorLike,
        control_point_flows: TensorLike,
        image_height: TensorLike,
        image_width: TensorLike,
        boundary_points_per_edge: TensorLike,
) -> tf.Tensor:
    """Add control points for zero-flow boundary conditions.

    Augment the set of control points with extra points on the
    boundary of the image that have zero flow.

    Args:
      control_point_locations: input control points.
      control_point_flows: their flows.
      image_height: image height.
      image_width: image width.
      boundary_points_per_edge: number of points to add in the middle of each
        edge (not including the corners).
        The total number of points added is
        `4 + 4*(boundary_points_per_edge)`.

    Returns:
      merged_control_point_locations: augmented set of control point locations.
      merged_control_point_flows: augmented set of control point flows.
    """

    batch_size = tf.compat.dimension_value(control_point_locations.shape[0])

    boundary_point_locations = _get_boundary_locations(
        image_height, image_width, boundary_points_per_edge
    )

    boundary_point_flows = np.zeros([boundary_point_locations.shape[0], 2])

    type_to_use = control_point_locations.dtype
    boundary_point_locations = tf.constant(
        _expand_to_minibatch(boundary_point_locations, batch_size), dtype=type_to_use
    )

    boundary_point_flows = tf.constant(
        _expand_to_minibatch(boundary_point_flows, batch_size), dtype=type_to_use
    )

    merged_control_point_locations = tf.concat(
        [control_point_locations, boundary_point_locations], 1
    )

    merged_control_point_flows = tf.concat(
        [control_point_flows, boundary_point_flows], 1
    )

    return merged_control_point_locations, merged_control_point_flows


def sparse_image_warp(
        image: TensorLike,
        source_control_point_locations: TensorLike,
        dest_control_point_locations: TensorLike,
        interpolation_order: int = 2,
        regularization_weight: FloatTensorLike = 0.0,
        num_boundary_points: int = 0,
        name: str = "sparse_image_warp",
) -> tf.Tensor:
    """Image warping using correspondences between sparse control points.

    Apply a non-linear warp to the image, where the warp is specified by
    the source and destination locations of a (potentially small) number of
    control points. First, we use a polyharmonic spline
    (`tfa.image.interpolate_spline`) to interpolate the displacements
    between the corresponding control points to a dense flow field.
    Then, we warp the image using this dense flow field
    (`tfa.image.dense_image_warp`).

    Let t index our control points. For `regularization_weight = 0`, we have:
    warped_image[b, dest_control_point_locations[b, t, 0],
                    dest_control_point_locations[b, t, 1], :] =
    image[b, source_control_point_locations[b, t, 0],
             source_control_point_locations[b, t, 1], :].

    For `regularization_weight > 0`, this condition is met approximately, since
    regularized interpolation trades off smoothness of the interpolant vs.
    reconstruction of the interpolant at the control points.
    See `tfa.image.interpolate_spline` for further documentation of the
    `interpolation_order` and `regularization_weight` arguments.


    Args:
      image: `[batch, height, width, channels]` float `Tensor`
      source_control_point_locations: `[batch, num_control_points, 2]` float
        `Tensor`
      dest_control_point_locations: `[batch, num_control_points, 2]` float
        `Tensor`
      interpolation_order: polynomial order used by the spline interpolation
      regularization_weight: weight on smoothness regularizer in interpolation
      num_boundary_points: How many zero-flow boundary points to include at
        each image edge. Usage:
          `num_boundary_points=0`: don't add zero-flow points
          `num_boundary_points=1`: 4 corners of the image
          `num_boundary_points=2`: 4 corners and one in the middle of each edge
            (8 points total)
          `num_boundary_points=n`: 4 corners and n-1 along each edge
      name: A name for the operation (optional).

      Note that image and offsets can be of type tf.half, tf.float32, or
      tf.float64, and do not necessarily have to be the same type.

    Returns:
      warped_image: `[batch, height, width, channels]` float `Tensor` with same
        type as input image.
      flow_field: `[batch, height, width, 2]` float `Tensor` containing the
        dense flow field produced by the interpolation.
    """

    image = tf.convert_to_tensor(image)
    source_control_point_locations = tf.convert_to_tensor(
        source_control_point_locations
    )
    dest_control_point_locations = tf.convert_to_tensor(dest_control_point_locations)

    control_point_flows = dest_control_point_locations - source_control_point_locations

    clamp_boundaries = num_boundary_points > 0
    boundary_points_per_edge = num_boundary_points - 1

    with tf.name_scope(name or "sparse_image_warp"):

        batch_size, image_height, image_width, _ = image.get_shape().as_list()

        # This generates the dense locations where the interpolant
        # will be evaluated.
        grid_locations = _get_grid_locations(image_height, image_width)

        flattened_grid_locations = np.reshape(
            grid_locations, [image_height * image_width, 2]
        )

        flattened_grid_locations = tf.constant(
            _expand_to_minibatch(flattened_grid_locations, batch_size), image.dtype
        )

        if clamp_boundaries:
            (
                dest_control_point_locations,
                control_point_flows,
            ) = _add_zero_flow_controls_at_boundary(
                dest_control_point_locations,
                control_point_flows,
                image_height,
                image_width,
                boundary_points_per_edge,
            )

        flattened_flows = interpolate_spline_ref(
            dest_control_point_locations,
            control_point_flows,
            flattened_grid_locations,
            interpolation_order,
            regularization_weight,
        )

        dense_flows = tf.reshape(
            flattened_flows, [batch_size, image_height, image_width, 2]
        )

        warped_image = dense_image_warp(image, dense_flows)

        return warped_image, dense_flows


def _cross_squared_distance_matrix(x: TensorLike, y: TensorLike) -> tf.Tensor:
    """Pairwise squared distance between two (batch) matrices' rows (2nd dim).

    Computes the pairwise distances between rows of x and rows of y.

    Args:
      x: `[batch_size, n, d]` float `Tensor`.
      y: `[batch_size, m, d]` float `Tensor`.

    Returns:
      squared_dists: `[batch_size, n, m]` float `Tensor`, where
      `squared_dists[b,i,j] = ||x[b,i,:] - y[b,j,:]||^2`.
    """
    x_norm_squared = tf.reduce_sum(tf.square(x), 2)
    y_norm_squared = tf.reduce_sum(tf.square(y), 2)

    # Expand so that we can broadcast.
    x_norm_squared_tile = tf.expand_dims(x_norm_squared, 2)
    y_norm_squared_tile = tf.expand_dims(y_norm_squared, 1)

    x_y_transpose = tf.matmul(x, y, adjoint_b=True)

    # squared_dists[b,i,j] = ||x_bi - y_bj||^2 =
    # x_bi'x_bi- 2x_bi'x_bj + x_bj'x_bj
    squared_dists = x_norm_squared_tile - 2 * x_y_transpose + y_norm_squared_tile

    return squared_dists


def _pairwise_squared_distance_matrix(x: TensorLike) -> tf.Tensor:
    """Pairwise squared distance among a (batch) matrix's rows (2nd dim).

    This saves a bit of computation vs. using
    `_cross_squared_distance_matrix(x, x)`

    Args:
      x: `[batch_size, n, d]` float `Tensor`.

    Returns:
      squared_dists: `[batch_size, n, n]` float `Tensor`, where
      `squared_dists[b,i,j] = ||x[b,i,:] - x[b,j,:]||^2`.
    """

    x_x_transpose = tf.matmul(x, x, adjoint_b=True)
    x_norm_squared = tf.linalg.diag_part(x_x_transpose)
    x_norm_squared_tile = tf.expand_dims(x_norm_squared, 2)

    # squared_dists[b,i,j] = ||x_bi - x_bj||^2 =
    # = x_bi'x_bi- 2x_bi'x_bj + x_bj'x_bj
    squared_dists = (
            x_norm_squared_tile
            - 2 * x_x_transpose
            + tf.transpose(x_norm_squared_tile, [0, 2, 1])
    )

    return squared_dists


def _solve_interpolation(
        train_points: TensorLike,
        train_values: TensorLike,
        order: int,
        regularization_weight: FloatTensorLike,
) -> TensorLike:
    r"""Solve for interpolation coefficients.

    Computes the coefficients of the polyharmonic interpolant for the
    'training' data defined by `(train_points, train_values)` using the kernel
    $\phi$.

    Args:
      train_points: `[b, n, d]` interpolation centers.
      train_values: `[b, n, k]` function values.
      order: order of the interpolation.
      regularization_weight: weight to place on smoothness regularization term.

    Returns:
      w: `[b, n, k]` weights on each interpolation center
      v: `[b, d, k]` weights on each input dimension
    Raises:
      ValueError: if d or k is not fully specified.
    """

    # These dimensions are set dynamically at runtime.
    b, n, _ = tf.unstack(tf.shape(train_points), num=3)

    d = train_points.shape[-1]
    if d is None:
        raise ValueError(
            "The dimensionality of the input points (d) must be "
            "statically-inferrable."
        )

    k = train_values.shape[-1]
    if k is None:
        raise ValueError(
            "The dimensionality of the output values (k) must be "
            "statically-inferrable."
        )

    # First, rename variables so that the notation (c, f, w, v, A, B, etc.)
    # follows https://en.wikipedia.org/wiki/Polyharmonic_spline.
    # To account for python style guidelines we use
    # matrix_a for A and matrix_b for B.

    c = train_points
    f = train_values

    # Next, construct the linear system.
    with tf.name_scope("construct_linear_system"):

        matrix_a = _phi(_pairwise_squared_distance_matrix(c), order)  # [b, n, n]
        if regularization_weight > 0:
            batch_identity_matrix = tf.expand_dims(tf.eye(n, dtype=c.dtype), 0)
            matrix_a += regularization_weight * batch_identity_matrix

        # Append ones to the feature values for the bias term
        # in the linear model.
        ones = tf.ones_like(c[..., :1], dtype=c.dtype)
        matrix_b = tf.concat([c, ones], 2)  # [b, n, d + 1]

        # [b, n + d + 1, n]
        left_block = tf.concat([matrix_a, tf.transpose(matrix_b, [0, 2, 1])], 1)

        num_b_cols = matrix_b.get_shape()[2]  # d + 1
        # -------------------------- Attention!! ----------------------------------#
        # lhs_zerosを0にすると動作しないので、0に限りなく近い値に置き換える
        lhs_zeros = tf.zeros([b, num_b_cols, num_b_cols], train_points.dtype)
        #lhs_zeros = tf.random.normal([b, num_b_cols, num_b_cols], mean=0.0, stddev=1.0, dtype=tf.float32) * EPSILON
        right_block = tf.concat([matrix_b, lhs_zeros], 1)  # [b, n + d + 1, d + 1]
        lhs = tf.concat([left_block, right_block], 2)  # [b, n + d + 1, n + d + 1]

        rhs_zeros = tf.zeros([b, d + 1, k], train_points.dtype)
        rhs = tf.concat([f, rhs_zeros], 1)  # [b, n + d + 1, k]

    # Then, solve the linear system and unpack the results.
    with tf.name_scope("solve_linear_system"):
        if tf.linalg.det(lhs) == 0:
            # TODO: Symptomatic treatment.
            #       Very rarely, when lhs becomes non-regular, regularization processing is performed.
            lhs += tf.expand_dims(tf.eye(lhs.shape[1], dtype=c.dtype), 0) * EPSILON
        w_v = tf.linalg.solve(lhs, rhs)
        w = w_v[:, :n, :]
        v = w_v[:, n:, :]

    return w, v


def _apply_interpolation(
        query_points: TensorLike,
        train_points: TensorLike,
        w: TensorLike,
        v: TensorLike,
        order: int,
) -> TensorLike:
    """Apply polyharmonic interpolation model to data.

    Given coefficients w and v for the interpolation model, we evaluate
    interpolated function values at query_points.

    Args:
      query_points: `[b, m, d]` x values to evaluate the interpolation at.
      train_points: `[b, n, d]` x values that act as the interpolation centers
          (the c variables in the wikipedia article).
      w: `[b, n, k]` weights on each interpolation center.
      v: `[b, d, k]` weights on each input dimension.
      order: order of the interpolation.

    Returns:
      Polyharmonic interpolation evaluated at points defined in `query_points`.
    """

    # First, compute the contribution from the rbf term.
    pairwise_dists = _cross_squared_distance_matrix(query_points, train_points)
    phi_pairwise_dists = _phi(pairwise_dists, order)

    rbf_term = tf.matmul(phi_pairwise_dists, w)

    # Then, compute the contribution from the linear term.
    # Pad query_points with ones, for the bias term in the linear model.
    query_points_pad = tf.concat(
        [query_points, tf.ones_like(query_points[..., :1], train_points.dtype)], 2
    )
    linear_term = tf.matmul(query_points_pad, v)

    return rbf_term + linear_term


def _phi(r: FloatTensorLike, order: int) -> FloatTensorLike:
    """Coordinate-wise nonlinearity used to define the order of the
    interpolation.

    See https://en.wikipedia.org/wiki/Polyharmonic_spline for the definition.

    Args:
      r: input op.
      order: interpolation order.

    Returns:
      `phi_k` evaluated coordinate-wise on `r`, for `k = r`.
    """

    # using EPSILON prevents log(0), sqrt0), etc.
    # sqrt(0) is well-defined, but its gradient is not
    with tf.name_scope("phi"):
        if order == 1:
            r = tf.maximum(r, EPSILON)
            r = tf.sqrt(r)
            return r
        elif order == 2:
            return 0.5 * r * tf.math.log(tf.maximum(r, EPSILON))
        elif order == 4:
            return 0.5 * tf.square(r) * tf.math.log(tf.maximum(r, EPSILON))
        elif order % 2 == 0:
            r = tf.maximum(r, EPSILON)
            return 0.5 * tf.pow(r, 0.5 * order) * tf.math.log(r)
        else:
            r = tf.maximum(r, EPSILON)
            return tf.pow(r, 0.5 * order)


def interpolate_spline_ref(
        train_points: TensorLike,
        train_values: TensorLike,
        query_points: TensorLike,
        order: int,
        regularization_weight: FloatTensorLike = 0.0,
        name: str = "interpolate_spline",
) -> tf.Tensor:
    r"""Interpolate signal using polyharmonic interpolation.

    The interpolant has the form
    $$f(x) = \sum_{i = 1}^n w_i \phi(||x - c_i||) + v^T x + b.$$

    This is a sum of two terms: (1) a weighted sum of radial basis function
    (RBF) terms, with the centers \\(c_1, ... c_n\\), and (2) a linear term
    with a bias. The \\(c_i\\) vectors are 'training' points.
    In the code, b is absorbed into v
    by appending 1 as a final dimension to x. The coefficients w and v are
    estimated such that the interpolant exactly fits the value of the function
    at the \\(c_i\\) points, the vector w is orthogonal to each \\(c_i\\),
    and the vector w sums to 0. With these constraints, the coefficients
    can be obtained by solving a linear system.

    \\(\phi\\) is an RBF, parametrized by an interpolation
    order. Using order=2 produces the well-known thin-plate spline.

    We also provide the option to perform regularized interpolation. Here, the
    interpolant is selected to trade off between the squared loss on the
    training data and a certain measure of its curvature
    ([details](https://en.wikipedia.org/wiki/Polyharmonic_spline)).
    Using a regularization weight greater than zero has the effect that the
    interpolant will no longer exactly fit the training data. However, it may
    be less vulnerable to overfitting, particularly for high-order
    interpolation.

    Note the interpolation procedure is differentiable with respect to all
    inputs besides the order parameter.

    We support dynamically-shaped inputs, where batch_size, n, and m are None
    at graph construction time. However, d and k must be known.

    Args:
      train_points: `[batch_size, n, d]` float `Tensor` of n d-dimensional
        locations. These do not need to be regularly-spaced.
      train_values: `[batch_size, n, k]` float `Tensor` of n c-dimensional
        values evaluated at train_points.
      query_points: `[batch_size, m, d]` `Tensor` of m d-dimensional locations
        where we will output the interpolant's values.
      order: order of the interpolation. Common values are 1 for
        \\(\phi(r) = r\\), 2 for \\(\phi(r) = r^2 * log(r)\\)
        (thin-plate spline), or 3 for \\(\phi(r) = r^3\\).
      regularization_weight: weight placed on the regularization term.
        This will depend substantially on the problem, and it should always be
        tuned. For many problems, it is reasonable to use no regularization.
        If using a non-zero value, we recommend a small value like 0.001.
      name: name prefix for ops created by this function

    Returns:
      `[b, m, k]` float `Tensor` of query values. We use train_points and
      train_values to perform polyharmonic interpolation. The query values are
      the values of the interpolant evaluated at the locations specified in
      query_points.
  """
    with tf.name_scope(name or "interpolate_spline"):
        train_points = tf.convert_to_tensor(train_points)
        train_values = tf.convert_to_tensor(train_values)
        query_points = tf.convert_to_tensor(query_points)

        # First, fit the spline to the observed data.
        with tf.name_scope("solve"):
            w, v = _solve_interpolation(
                train_points, train_values, order, regularization_weight
            )

        # Then, evaluate the spline at the query locations.
        with tf.name_scope("predict"):
            query_values = _apply_interpolation(query_points, train_points, w, v, order)

    return query_values
