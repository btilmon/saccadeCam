from __future__ import absolute_import, division, print_function

from trainer_main import Trainer
from options import Options

options = Options()
opts = options.parse()

if __name__ == "__main__":

    trainer = Trainer(opts)
    trainer.train()
        
