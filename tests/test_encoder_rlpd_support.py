import jax
import jax.numpy as jnp

from flow_policy.encoder_rlpd import _latent_support_constraint


def test_nonstandard_actor_distribution_can_be_contained() -> None:
    mean = jnp.asarray([[0.75, -0.5]])
    std = jnp.asarray([[0.5, 0.6]])

    overflow, violation, fraction, lower, upper = _latent_support_constraint(
        mean, std, prior_radius=3.0, policy_stddevs=3.0, tolerance=0.0
    )

    assert float(overflow) == 0.0
    assert float(violation) == 0.0
    assert float(fraction) == 1.0
    assert float(lower) >= -3.0
    assert float(upper) <= 3.0


def test_support_overflow_penalizes_mean_and_scale() -> None:
    mean = jnp.asarray([[2.0]])
    std = jnp.asarray([[0.5]])

    overflow, violation, fraction, _, upper = _latent_support_constraint(
        mean, std, prior_radius=3.0, policy_stddevs=3.0, tolerance=0.0
    )
    mean_grad, std_grad = jax.grad(
        lambda actor_mean, actor_std: _latent_support_constraint(
            actor_mean,
            actor_std,
            prior_radius=3.0,
            policy_stddevs=3.0,
            tolerance=0.0,
        )[1],
        argnums=(0, 1),
    )(mean, std)

    assert float(overflow) > 0.0
    assert float(violation) > 0.0
    assert float(fraction) == 0.0
    assert float(upper) > 3.0
    assert float(mean_grad[0, 0]) > 0.0
    assert float(std_grad[0, 0]) > 0.0
