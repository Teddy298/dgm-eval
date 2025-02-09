import jax
import jax.numpy as jnp

Array = jnp.ndarray

@jax.jit
def energy(x, y):
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
  n_x = x.shape[0]
  y = jnp.asarray(y)
  n_y = y.shape[0]

  # jnp.matmul(x, x.T) etc. are not cached to avoid OOM when x has many rows.
  x_sqnorms = jnp.diag(jnp.matmul(x, x.T))
  y_sqnorms = jnp.diag(jnp.matmul(y, y.T))

  k_xx = n_x/(n_x-1) * jnp.mean(
      jnp.sqrt(
              -2 * jnp.matmul(x, x.T)
              + jnp.expand_dims(x_sqnorms, 1)
              + jnp.expand_dims(x_sqnorms, 0)

      )
  )
  k_xy = jnp.mean(
      jnp.sqrt(
              -2 * jnp.matmul(x, y.T)
              + jnp.expand_dims(x_sqnorms, 1)
              + jnp.expand_dims(y_sqnorms, 0)
      )
  )
  k_yy = n_y/(n_y-1) * jnp.mean(
      jnp.sqrt(
              -2 * jnp.matmul(y, y.T)
              + jnp.expand_dims(y_sqnorms, 1)
              + jnp.expand_dims(y_sqnorms, 0)
      )
  )

  return (2 * k_xy - k_xx - k_yy)/(2 * k_xy)