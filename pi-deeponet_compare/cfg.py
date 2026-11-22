from dataclasses import dataclass, field, replace
from typing import Tuple


@dataclass(frozen=True)
class GridConfig:
    half: float = 0.5
    M: int = 10

    @property
    def n(self) -> int:
        return 2 * self.M + 1

    @property
    def h(self) -> float:
        return 1.0 / (2 * self.M)


@dataclass(frozen=True)
class DataConfig:
    family: str = "a_q"
    c_scale: float = 12.0
    n_samples: int = 1000
    n_test: int = 100
    seed: int = 0

    s_low: float = 0.7
    s_high: float = 1.3
    mu_low: float = 5.0
    mu_high: float = 10.0

    A0: float = 0.40
    C2: float = 1.50
    L0: float = 0.10
    L1: float = 0.10

    cache: bool = True

    @property
    def n_train(self) -> int:
        return self.n_samples - self.n_test


@dataclass(frozen=True)
class ModelConfig:
    branch_kind: str = "mlp"
    branch_layers: Tuple[int, ...] = (128, 128, 128)
    branch_activation: str = "relu"
    trunk_layers: Tuple[int, ...] = (128, 128, 128)
    activation: str = "tanh"
    feature_dim: int = 128
    branch_out_scale: float = 0.01
    fourier_m: int = 0
    fourier_sigma: float = 10.0


    input_scale_a: float = 1.0
    input_scale_lam: float = 1.0
    input_scale_c: float = 10.0


@dataclass(frozen=True)
class PhysicsConfig:
    n_colloc: int = 512
    n_data: int = 20
    margin: float = 0.02


    bc_mode: str = "adf_aq"
    n_bc: int = 512
    w_bc_anti: float = 1.0
    w_bc_diag: float = 10.0

    pde_norm: str = "terms"
    colloc_margin: float = 0.02

    w_pde: float = 1.0
    w_data: float = 10.0

    fixed_colloc: bool = False
    interior_anchors: bool = True


    online_truth: bool = True


    positive_x: bool = False
    negative_x: bool = False


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 100
    batch_size: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 0
    log_every: int = 20
    eval_every: int = 100
    eval_samples: int = 20
    warmup_steps: int = 2
    lr_schedule: str = "cosine"
    clip_norm: float = 1.0


@dataclass
class Config:
    grid: GridConfig = field(default_factory=GridConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def copy(self, **kwargs) -> "Config":
        return replace(self, **kwargs)


MODE_DEFAULTS = {
    "deeponet": dict(branch_kind="mlp", feature_dim=128,
                     trunk_layers=(128, 128, 128)),
    "pi-deeponet": dict(branch_kind="mlp", feature_dim=128,
                        trunk_layers=(128, 128, 128)),
    "sep-pi-deeponet": dict(branch_kind="mlp", feature_dim=128,
                            trunk_layers=(128, 128, 128)),
}


def config_for_mode(mode: str, **overrides) -> Config:
    if mode not in MODE_DEFAULTS:
        raise ValueError(f"unknown mode {mode!r}; choose from {list(MODE_DEFAULTS)}")
    mcfg = replace(ModelConfig(), **MODE_DEFAULTS[mode])
    cfg = Config(model=mcfg)
    groups = {
        "grid": (GridConfig, {}),
        "data": (DataConfig, {}),
        "model": (ModelConfig, {}),
        "physics": (PhysicsConfig, {}),
        "train": (TrainConfig, {}),
    }
    for key, value in overrides.items():
        if value is None:
            continue
        for name, (cls, kw) in groups.items():
            if key in cls.__dataclass_fields__:
                kw[key] = value
                break
        else:
            raise TypeError(f"unknown config key {key!r}")
    return replace(
        cfg,
        grid=replace(cfg.grid, **groups["grid"][1]),
        data=replace(cfg.data, **groups["data"][1]),
        model=replace(cfg.model, **groups["model"][1]),
        physics=replace(cfg.physics, **groups["physics"][1]),
        train=replace(cfg.train, **groups["train"][1]),
    )

