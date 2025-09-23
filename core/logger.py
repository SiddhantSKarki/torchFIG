import os
from torch.utils.tensorboard import SummaryWriter

class Logger:
    def __init__(self, cfg):
        self.writer = None
        if cfg["logging"]["use_tensorboard"]:
            log_dir = os.path.join(cfg["logging"]["log_dir"], cfg["experiment"]["name"])
            self.writer = SummaryWriter(log_dir=log_dir)
            print(f"[Logger] TensorBoard at {log_dir}")
        else:
            raise ValueError(f"Error: `use_tensorboard` attribute set to {cfg['logging']['use_tensorboard']}")

    def add_scalar(self, tag, value, step):
        if self.writer: self.writer.add_scalar(tag, value, step)

    # TODO: Need to implement image logging
    def add_image(self, tag, value, step):
        if self.writer: self.writer.add_s(tag, value, step)

    def add_histogram(self, tag, values, step):
        if self.writer: self.writer.add_histogram(tag, values, step)

    def close(self):
        if self.writer: self.writer.close()
