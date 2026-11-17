# SPDX-FileCopyrightText: 2023-2026 Jakob Schatzl
# SPDX-License-Identifier: Apache-2.0

# @file learnrate.py
# @author Jakob Schatzl
# @brief Learning rate schedules for training
# @version 1.0
# @date 2026-11-17

def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


def step_decay_lr(
    epoch: int,
    initial_lr: float,
    step_size: int,
    gamma: float = 0.1,
) -> float:
    decay_steps = epoch // step_size
    lr = initial_lr * gamma**decay_steps
    return lr


def linear_decay_lr(
    epoch: int,
    initial_lr: float,
    final_lr: float,
    decay_epochs: int,
) -> float:
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
    power: float = 3.0,
) -> float:
    normalized_error = min(error / reference_error, 1.0)
    scaled_error = normalized_error**power

    return min_lr + (max_lr - min_lr) * scaled_error