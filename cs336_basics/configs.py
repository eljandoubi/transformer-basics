from pathlib import Path
from dataclasses import dataclass, asdict, field
import torch


class BaseConfig:
    def to_dict(self):
        return asdict(self)


@dataclass
class ModelConfig(BaseConfig):
    vocab_size: int = 512  # 32768
    context_length: int = 4  # 2048
    d_model: int = 4  # 1024
    num_heads: int = 2
    num_layers: int | None = None
    d_ff: int | None = None
    rope_theta: float = 10000.0

    def __post_init__(self):
        if self.d_ff is None:
            self.d_ff = 4 * self.d_model

        if self.num_layers is None:
            self.num_layers = max(self.d_model // 16, 1)  # 128


@dataclass
class AdamWConfig(BaseConfig):
    lr: float = 1e-3
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8


@dataclass
class CosineSchedulerConfig(BaseConfig):
    min_learning_rate: float = 1e-6
    warmup_iters: int = 1
    cosine_cycle_iters: int | None = None
    last_epoch: int = -1
    verbose: str = "deprecated"


@dataclass
class TrainerConfig(BaseConfig):
    epochs: int = 10
    train_batch_size: int = 2
    val_batch_size: int = 4
    checkpoint_folder: str | Path = "./checkpointing"
    checkpoint_path: str | Path | None = None
    gradient_accumulation: int = 1

    def __post_init__(self):
        self.checkpoint_folder = Path(self.checkpoint_folder)
        self.checkpoint_folder.mkdir(parents=True, exist_ok=True)
        if self.checkpoint_path is None:
            self.checkpoint_path = "checkpoint.pt"
        self.checkpoint_path = self.checkpoint_folder / self.checkpoint_path


@dataclass
class TokenizerConfig(BaseConfig):
    vocab_path: str = "./tokenizer/gpt2_vocab.json"
    merges_path: str = "./tokenizer/gpt2_merges.txt"
    special_tokens_path: str = "./tokenizer/special.txt"


@dataclass
class End2EndConfig(BaseConfig):
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: AdamWConfig = field(default_factory=AdamWConfig)
    sched: CosineSchedulerConfig = field(default_factory=CosineSchedulerConfig)
    tokens: TokenizerConfig = field(default_factory=TokenizerConfig)
    device: str | None = None
    seed: int = 42

    def __post_init__(self):
        if self.device is None:
            if torch.cuda.is_available():
                index = torch.cuda.current_device()
                self.device = f"cuda:{index}"
            else:
                self.device = "cpu"
        if self.sched.cosine_cycle_iters is None:
            self.sched.cosine_cycle_iters = self.trainer.epochs * 2 // 3


if __name__ == "__main__":
    print(End2EndConfig())
