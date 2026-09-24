from math import floor
from typing import Optional


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


def step_decay_lr(
    epoch: int,
    initial_lr: float,
    step_size: int,
    gamma: float = 0.1,
) -> float:
    """
    Step-decay learning-rate schedule.

    Parameters
    ----------
    epoch:
        Current epoch, starting at 0.
    initial_lr:
        Initial learning rate.
    step_size:
        Number of epochs between decays.
    gamma:
        Multiplicative decay factor.

    Returns
    -------
    float
        Learning rate for the current epoch.
    """
    if epoch < 0:
        raise ValueError("epoch must be >= 0")
    if initial_lr <= 0:
        raise ValueError("initial_lr must be > 0")
    if step_size <= 0:
        raise ValueError("step_size must be > 0")
    if not 0 < gamma <= 1:
        raise ValueError("gamma must be in the interval (0, 1]")

    decay_steps = epoch // step_size
    return initial_lr * gamma**decay_steps


def linear_decay_lr(
    epoch: int,
    initial_lr: float,
    final_lr: float,
    decay_epochs: int,
) -> float:
    """
    Linear learning-rate decay.

    The learning rate decreases linearly from initial_lr to final_lr
    over decay_epochs epochs and remains at final_lr afterward.

    Parameters
    ----------
    epoch:
        Current epoch, starting at 0.
    initial_lr:
        Learning rate at epoch 0.
    final_lr:
        Learning rate reached at decay_epochs.
    decay_epochs:
        Number of epochs over which to decay.

    Returns
    -------
    float
        Learning rate for the current epoch.
    """
    if epoch < 0:
        raise ValueError("epoch must be >= 0")
    if initial_lr <= 0:
        raise ValueError("initial_lr must be > 0")
    if final_lr < 0:
        raise ValueError("final_lr must be >= 0")
    if final_lr > initial_lr:
        raise ValueError("final_lr must not be greater than initial_lr")
    if decay_epochs <= 0:
        raise ValueError("decay_epochs must be > 0")

    progress = min(epoch / decay_epochs, 1.0)

    return initial_lr + progress * (final_lr - initial_lr)


def lr_inverse_time(
    step: int,
    *,
    lr_initial: float,
    decay: float,
    lr_min: float = 0.0,
) -> float:
    """
    Inverse-time decay:

        lr = lr_initial / (1 + decay * step)
    """
    if step < 0:
        raise ValueError("step must be non-negative")
    if decay < 0.0:
        raise ValueError("decay must be non-negative")
    if lr_initial < 0.0 or lr_min < 0.0:
        raise ValueError("learning rates must be non-negative")

    return max(lr_min, lr_initial / (1.0 + decay * step))


def error_dependent_lr(
    error: float,
    min_lr: float,
    max_lr: float,
    reference_error: float,
    power: float = 1.0,
) -> float:
    """
    Error-dependent learning rate.

    The learning rate is high when the error is large and approaches
    min_lr as the error approaches zero.

    Parameters
    ----------
    error:
        Current error or loss. Usually this should be non-negative.
    min_lr:
        Minimum learning rate.
    max_lr:
        Maximum learning rate.
    reference_error:
        Error corresponding to max_lr.
    power:
        Controls the curve shape:
          power = 1.0 -> linear
          power > 1.0 -> lower LR for moderate errors
          power < 1.0 -> higher LR for moderate errors

    Returns
    -------
    float
        Learning rate based on the current error.
    """
    if error < 0:
        raise ValueError("error must be >= 0")
    if min_lr <= 0:
        raise ValueError("min_lr must be > 0")
    if max_lr < min_lr:
        raise ValueError("max_lr must be >= min_lr")
    if reference_error <= 0:
        raise ValueError("reference_error must be > 0")
    if power <= 0:
        raise ValueError("power must be > 0")

    normalized_error = min(error / reference_error, 1.0)
    scaled_error = normalized_error**power

    return min_lr + (max_lr - min_lr) * scaled_error