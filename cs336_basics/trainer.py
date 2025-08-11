import logging
import itertools
import torch
from tqdm import tqdm
from cs336_basics.configs import End2EndConfig
from cs336_basics.tokenizer import BPETokenizer
from cs336_basics.model import TransformerDecoder
from cs336_basics.optimizer import AdamW, CosineScheduler
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.utils import load_checkpoint, save_checkpoint, set_seed, get_batch


logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")


class Pipeline:
    def __init__(self, config: End2EndConfig = End2EndConfig()):
        logging.info("Initialize Training Pipeline")
        set_seed(config.seed)
        self.config = config
        self.step: int = 0
        self.model = TransformerDecoder(**config.model.to_dict())
        self.prepare_model()
        self.optim = AdamW(self.model.parameters(), **config.optim.to_dict())
        self.sched = CosineScheduler(self.optim, **config.sched.to_dict())
        self.token = BPETokenizer.from_vocab(**config.tokens.to_dict())
        self.load_state()

    def prepare_model(self):
        logging.info("Prepare Model.")
        logging.info("Move model to %s", self.config.device)
        self.model = self.model.to(self.config.device)
        logging.info("Tie token embedding and LM head weights.")
        self.model.tie_weights()

    def train(self, train_data_path: str, val_data_path: str):
        logging.info("Start Training")
        bs = self.config.trainer.val_batch_size
        cl = self.config.model.context_length
        gr_acc = self.config.trainer.gradient_accumulation
        epochs = self.config.trainer.epochs
        device = self.config.device
        self.model.train()
        raw_data = self.token.stream_sequence(train_data_path)
        len_data = sum(len(seq) for seq in tqdm(raw_data, "Count tokens"))
        logging.info("token count %s", len_data)
        bar = tqdm(desc="Train step", total=(len_data * epochs) // (bs * (cl + 1)))
        for e in range(1, epochs + 1):
            log_perplexity = torch.zeros(1, device=self.config.device)
            raw_data = self.token.stream_sequence(train_data_path)
            for i in range(1, len_data // (bs * (cl + 1)) + 1):
                seq = next(raw_data)
                if (e * i) < self.step or len(seq) < cl:
                    continue
                x, y = get_batch(seq, bs, cl, device)
                self.optim.zero_grad()
                loss = cross_entropy(self.model(x), y)
                loss.backward()
                log_perplexity += loss.detach()
                if (i * e) % gr_acc == 0:
                    self.optim.step()
                    self.sched.step()
                    self.step = e * i

                bar.update()
            self.save_state()
            train_perplexity = bs * log_perplexity.item()
            logging.info("Train log perplexity at epoch %s: %s", e, train_perplexity)
            val_perplexity = self.valid(val_data_path)
            logging.info("Valid log perplexity at epoch %s: %s", e, val_perplexity)

        bar.close()

    @torch.inference_mode()
    def valid(self, val_data_path: str, stream: bool = False) -> float:
        logging.info("Start Validation")
        bs = self.config.trainer.val_batch_size
        cl = self.config.model.context_length
        self.model.eval()
        if stream:
            raw_data = self.token.stream_encode(val_data_path)
        else:
            raw_data = self.token.encode_file(val_data_path)

        log_perplexity = torch.zeros(1, device=self.config.device)
        sample = []
        tmp, raw_data = itertools.tee(raw_data)
        bar = tqdm(desc="Eval step", total=sum(1 for _ in tmp) // (bs * (cl + 1)))
        del tmp
        for token in raw_data:
            sample.append(token)
            if len(sample) == bs * (cl + 1):
                bar.update()
                xy = torch.tensor(sample, pin_memory=True).reshape((bs, cl + 1))
                if "cuda" in self.config.device:
                    xy = xy.to(device=self.config.device, non_blocking=True)
                x, y = xy[:, :-1], xy[:, 1:]
                sample = []
                log_perplexity += cross_entropy(self.model(x), y)
        log_perplexity *= bs
        bar.close()
        return log_perplexity.item()

    def save_state(self):
        logging.info("Save the state at step %s to %s", self.step, self.config.trainer.checkpoint_path)
        save_checkpoint(self.model, self.optim, self.step, self.config.trainer.checkpoint_path)

    def load_state(self):
        if self.config.trainer.checkpoint_path.exists():
            logging.info("Load state from %s", self.config.trainer.checkpoint_path)
            self.step = load_checkpoint(self.config.trainer.checkpoint_path, self.model, self.optim)
        else:
            logging.warning("There is no such file: %s", self.config.trainer.checkpoint_path)

    @torch.no_grad()
    def check_forward(self):
        logging.info("Check Forward")
        x = torch.randint(
            0,
            self.config.model.vocab_size,
            (self.config.trainer.batch_size, self.config.model.context_length),
            device=self.config.device,
        )
        return self.model(x)


if __name__ == "__main__":
    pipe = Pipeline()
    pipe.train("data/TinyStoriesV2-GPT4-valid.txt", "data/TinyStoriesV2-GPT4-valid.txt")
