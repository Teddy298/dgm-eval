# coding=utf-8
# Copyright 2024 The Google Research Authors.
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

"""A memory-efficient MMD implementation in JAX."""

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

Array = jnp.ndarray

# The bandwidth parameter for the Gaussian RBF kernel. See the paper for more
# details.
_SIGMA = 10
# The following is used to make the metric more human readable. See the paper
# for more details.
_SCALE = 1


@jax.jit
def cmmd(x, y):
  """A memory-efficient MMD implementation in JAX.

  This implements the minimum-variance/biased version of the estimator described
  in Eq.(5) of
  https://jmlr.csail.mit.edu/papers/volume13/gretton12a/gretton12a.pdf.
  As described in Lemma 6's proof in that paper, the unbiased estimate and the
  minimum-variance estimate for MMD are almost identical.

  Note that the first invocation of this function will be considerably slow due
  to JAX JIT compilation.

  Args:
    x: The first set of embeddings of shape (n, embedding_dim).
    y: The second set of embeddings of shape (n, embedding_dim).

  Returns:
    The MMD distance between x and y embedding sets.
  """
  x = jnp.asarray(x)
  y = jnp.asarray(y)

  # jnp.matmul(x, x.T) etc. are not cached to avoid OOM when x has many rows.
  x_sqnorms = jnp.diag(jnp.matmul(x, x.T))
  y_sqnorms = jnp.diag(jnp.matmul(y, y.T))

  gamma = 1 / (2 * _SIGMA**2)
  k_xx_logsumexp = jnp.exp(
      logsumexp(
          -gamma
          * (
                  -2 * jnp.matmul(x, x.T)
                  + jnp.expand_dims(x_sqnorms, 1)
                  + jnp.expand_dims(x_sqnorms, 0)
          ),
          axis=(0, 1)  # Summing over both matrix dimensions
      ) - jnp.log(x.shape[0] * x.shape[0])  # Normalize properly
  )

  k_xy_logsumexp = jnp.exp(
      logsumexp(
          -gamma
          * (
                  -2 * jnp.matmul(x, y.T)
                  + jnp.expand_dims(x_sqnorms, 1)
                  + jnp.expand_dims(y_sqnorms, 0)
          ),
          axis=(0, 1)  # Sum over both matrix dimensions
      ) - jnp.log(x.shape[0] * y.shape[0])  # Correct normalization
  )
  k_yy_logsumexp = jnp.exp(
      logsumexp(
          -gamma
          * (
                  -2 * jnp.matmul(y, y.T)
                  + jnp.expand_dims(y_sqnorms, 1)
                  + jnp.expand_dims(y_sqnorms, 0)
          ),
          axis=(0, 1)  # Summing over both matrix dimensions
      ) - jnp.log(y.shape[0] * y.shape[0])  # Normalize properly
  )
  all_components = {
          "k_xx": k_xx_logsumexp,
          "k_yy": k_yy_logsumexp,
          "k_xy": k_xy_logsumexp
          }
  return k_xx_logsumexp + k_yy_logsumexp - 2 * k_xy_logsumexp, k_xx_logsumexp + k_yy_logsumexp, all_components

_BLOCK_SIZE = 1000  # Hardcoded block size

@jax.jit
def blockwise_kernel_mean(x, y):
    """Computes the mean of the kernel function in a blockwise manner without constructing full matrices."""
    n = x.shape[0]
    num_blocks = n // _BLOCK_SIZE  # Ensure divisibility for simplicity
    gamma = 1 / (2 * _SIGMA**2)

    x_sqnorms = jnp.diag(jnp.matmul(x, x.T))
    y_sqnorms = jnp.diag(jnp.matmul(y, y.T))

    def block_kernel_mean(i, mean_accum):
        row_start = (i // num_blocks) * _BLOCK_SIZE
        col_start = (i % num_blocks) * _BLOCK_SIZE

        # Slice blocks
        x_block = jax.lax.dynamic_slice(x, (row_start, 0), (_BLOCK_SIZE, x.shape[1]))
        y_block = jax.lax.dynamic_slice(y, (col_start, 0), (_BLOCK_SIZE, y.shape[1]))

        # Compute squared norms for blocks
        x_sq_block = jnp.diag(jnp.matmul(x_block, x_block.T))
        y_sq_block = jnp.diag(jnp.matmul(y_block, y_block.T))

        # Compute kernel matrix block
        k_block = jnp.exp(
            -gamma
            * (
                -2 * jnp.matmul(x_block, y_block.T)
                + jnp.expand_dims(x_sq_block, 1)
                + jnp.expand_dims(y_sq_block, 0)
            )
        )

        return mean_accum + jnp.mean(k_block)

    mean_sum = jax.lax.fori_loop(0, num_blocks**2, block_kernel_mean, 0.0)
    return mean_sum / (num_blocks**2)

@jax.jit
def cmmd_blockwise(x, y):
    """Computes C-MMD using blockwise kernel computation."""
    mean_kxx = blockwise_kernel_mean(x, x)
    mean_kxy = blockwise_kernel_mean(x, y)
    mean_kyy = blockwise_kernel_mean(y, y)

    return _SCALE * (mean_kxx + mean_kyy - 2 * mean_kxy)
