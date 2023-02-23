import numpy as np
import tensorflow as tf
import scipy.signal


def get_flat(nested_tensor):
    """get flattened tensor"""
    flattened = tf.concat(
        [tf.reshape(t, [-1]) for t in nested_tensor],
        axis=0,
    )
    return flattened


def set_from_flat(model, flat_weights):
    """
    Create the process of assigning updated vars
    """
    """set model weights from flattened grads"""
    weights = []
    idx = 0
    for var in model.trainable_variables:
        n_vars = tf.reduce_prod(var.shape)
        var_weight = tf.reshape(flat_weights[idx: idx + n_vars], var.shape)
        var.assign(var_weight)
        # weights.append(tf.reshape(flat_weights[idx: idx + n_vars], var.shape))
        idx += n_vars
    # model.set_weights(weights)


def linesearch(model, fn, theta_k, fullstep, expected_improve_rate, data):
    accept_ratio = .1
    max_backtracks = 10

    _, action_logprob, _ = fn(data['states'], data['encodes_z'], data['actions'])
    ratio = tf.exp(action_logprob - data['old_action_logprob'])
    surr = tf.reduce_mean(tf.math.multiply(ratio, data['advants']))
    theta_updated = False
    for _n_backtracks in tf.range(max_backtracks):  # alpha=0.5
        step_size = tf.math.pow(0.5, tf.cast(_n_backtracks, dtype=tf.float32))
        # Compute new theta
        theta_new = theta_k + step_size * fullstep

        # Set the new weights and compute the surrogate loss again
        set_from_flat(model, theta_new)
        _, new_action_logprob, _ = fn(data['states'], data['encodes_z'], data['actions'])
        new_ratio = tf.exp(new_action_logprob - data['old_action_logprob'])
        new_surr = tf.reduce_mean(tf.math.multiply(new_ratio, data['advants']))

        # Check the improvement in surrogate loss
        actual_improve = new_surr - surr
        expected_improve = expected_improve_rate * step_size
        ratio = actual_improve / expected_improve
        if ratio > accept_ratio and actual_improve > 0:  # Todo: Also add condition to check when kl div is violated
            theta_updated = True
            break

    if theta_updated:
        return True, theta_new
    else:
        return False, theta_k


def conjugate_gradient(f_Ax, b_vec, cg_iters, residual_tol=1e-10):  # Do not play with residual_tol
    x_var = tf.zeros_like(b_vec)  # Initialise X0 <- can be zeros
    first_basis_vect = tf.identity(b_vec)  # Initialise Conjugate direction P0 same as r0=b
    residual = tf.identity(b_vec)  # Residual, r0 = b - A*x0 which equals b given x0 = 0
    rdotr = tf.reduce_sum(residual*residual)  # L2 norm of the residual
    for _ in tf.range(cg_iters):  # Theoretically, the method converges in n=dim(b) iterations
        z_var = f_Ax(first_basis_vect)  # Compute vector product AxP
        alpha = rdotr / tf.reduce_sum(first_basis_vect * z_var)
        x_var += alpha * first_basis_vect  # Update approx of x* that solves Ax=b
        residual -= alpha * z_var  # Update residuals
        new_rdotr = tf.reduce_sum(residual*residual)  # Compute <r_(k+1), r_(k+1)>
        beta = new_rdotr / rdotr
        first_basis_vect = residual + beta * first_basis_vect  # Get the next conjugate direction to move along
        rdotr = new_rdotr
        if rdotr < residual_tol:  # If r = b-Ax -> 0 we are close to x*
            break
    return x_var
