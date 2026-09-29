"""Separate data-split, DRE cross-fit, and model randomness without changing defaults."""


def resolve_seed(config, component, *, default_seed=None):
    if component not in {"split", "dre_fold", "model"}:
        raise ValueError(f"Unknown randomness component: {component}")
    value = config.get("randomness", {}).get(f"{component}_seed")
    if value is None:
        value = config["seed"] if default_seed is None else default_seed
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**31 - 100:
        raise ValueError(f"{component}_seed must be a nonnegative 32-bit integer")
    return value


def resolved_seeds(config):
    return {f"{component}_seed": resolve_seed(config, component)
            for component in ["split", "dre_fold", "model"]}
