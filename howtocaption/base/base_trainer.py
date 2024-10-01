import torch
from abc import abstractmethod
from numpy import inf
import os
import sys
from collections import OrderedDict

from howtocaption.utils.dist_utils import is_main_process


class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, model, model_without_ddp, loss, metrics, optimizer, device, config,
                 epochs, save_period, monitor='off', init_val=False, early_stop=inf, neptune_run=None,
                 resume_only_model=False, resume_only_model_and_opt=False,
                 freq_eval=1, nlp_freq_eval=10, retrieval_freq_eval=10000000,
                 freq_visual_input=50, log_visual_input_at_start=True,
                 init_nlp=False, init_retrieval=False, init_val_loss=False, save_epochs=None, remove_resume=False, load_strict=True):

        self.model = model
        self.model_without_ddp = model_without_ddp
        self.loss = loss
        self.metrics = metrics
        self.optimizer = optimizer
        self.device = device

        self.config = config

        self.epochs = epochs
        self.save_period = save_period
        self.save_epochs = save_epochs
        if self.save_epochs is not None:
            assert isinstance(self.save_epochs, list)
        self.monitor = monitor
        self.init_val = init_val
        self.init_nlp = init_nlp
        self.init_retrieval = init_retrieval
        self.init_val_loss = init_val_loss

        self.resume_only_model = resume_only_model
        self.resume_only_model_and_opt = resume_only_model_and_opt
        self.remove_resume = remove_resume
        self.load_strict = load_strict

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']
            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = early_stop
            if self.early_stop <= 0:
                self.early_stop = inf

        self.start_epoch = 1
        self.step = 0

        self.checkpoint_dir = config.save_dir

        # # setup visualization writer instance
        # self.writer = TensorboardWriter(config.log_dir, self.logger, cfg_trainer['tensorboard'])
        self.neptune_run = neptune_run

        # how often to evaluate
        self.freq_eval = freq_eval

        # how often to evaluate all these caption metrics
        self.nlp_freq_eval = nlp_freq_eval

        # how often to retrieval
        self.retrieval_freq_eval = retrieval_freq_eval

        # how often to log visual input
        self.freq_visual_input = freq_visual_input
        self.log_visual_input_at_start = log_visual_input_at_start

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    @abstractmethod
    def _valid_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    @abstractmethod
    def _eval_nlp(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError


    @abstractmethod
    def _eval_retrieval(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """
        not_improved_count = 0

        # self._eval_retrieval(self.start_epoch - 1)


        if self.init_val:
            log = self._valid_epoch(self.start_epoch - 1)
            # print logged informations to the screen
            for key, value in log.items():
                print('    {:15s}: {}'.format(str(key), value))

        if self.init_retrieval:
            self._eval_retrieval(self.start_epoch - 1)

        if self.init_nlp:
            self._eval_nlp(self.start_epoch - 1)

        for epoch in range(self.start_epoch, self.epochs + 1):
            # with torch.autograd.set_detect_anomaly(True):
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            for key, value in result.items():
                if key == 'metrics':
                    log.update({mtr.__name__: value[i]
                                for i, mtr in enumerate(self.metrics)})
                elif key == 'val_metrics':
                    log.update({'val_' + mtr.__name__: value[i]
                                for i, mtr in enumerate(self.metrics)})
                elif key == 'nested_val_metrics':
                    # NOTE: currently only supports two layers of nesting
                    for subkey, subval in value.items():
                        for subsubkey, subsubval in subval.items():
                            for subsubsubkey, subsubsubval in subsubval.items():
                                log[f"val_{subkey}_{subsubkey}_{subsubsubkey}"] = subsubsubval
                else:
                    log[key] = value

            # print logged informations to the screen
            for key, value in log.items():
                print('    {:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    print("Warning: Metric '{}' is not found. "
                                        "Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    print("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))
                    break

            if best:
                self._save_checkpoint(epoch, save_best=best)

            if epoch % self.save_period == 0 :
                self._save_checkpoint(epoch)

            if (self.save_epochs is not None) and (epoch in self.save_epochs):
                self._save_checkpoint(epoch)

    def _save_checkpoint(self, epoch, save_best=False, save_latest=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """

        if not is_main_process():  # in case of distributed training
            return

        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'step': self.step,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'loss': self.loss.state_dict() if self.loss is not None else None,
            'monitor_best': self.mnt_best,
            'config': self.config.config
        }

        if save_latest:
            # for safety
            tmp_best_path = str(self.checkpoint_dir / 'tmp.pth')
            torch.save(state, tmp_best_path, _use_new_zipfile_serialization=False)
            best_path = str(self.checkpoint_dir / 'latest_model.pth')
            os.rename(tmp_best_path, best_path)
            print("Saving current model: latest_model.pth ...") # safe in terms of "No space left on device"

        if save_best:
            tmp_best_path = str(self.checkpoint_dir / 'tmp.pth')  # safe in terms of "No space left on device"
            torch.save(state, tmp_best_path, _use_new_zipfile_serialization=False)
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            os.rename(tmp_best_path, best_path)
            print("Saving current best: model_best.pth ...")

        if not(save_best or save_latest):
            filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
            torch.save(state, filename, _use_new_zipfile_serialization=False)
            print("Saving checkpoint: {} ...".format(filename))

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        print("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path, map_location='cpu')
        print("Loading from epoch: {} ...".format(checkpoint['epoch']), flush=True)

        if not (self.resume_only_model or self.resume_only_model_and_opt):
            self.start_epoch = checkpoint['epoch'] + 1
            self.step = checkpoint['step'] + 1
            self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        state_dict = checkpoint['state_dict']
        state_dict = fix_module_in_state_dict(state_dict, self.model)
        self.model.load_state_dict(state_dict, strict=self.load_strict)

        if not self.resume_only_model:
            # load optimizer state from checkpoint only when optimizer type is not changed.
            if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
                print("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                    "Optimizer parameters not being resumed.")
            else:
                self.optimizer.load_state_dict(checkpoint['optimizer'])

        if self.remove_resume:
            assert self.resume_only_model == False
            assert self.resume_only_model_and_opt == False

            os.unlink(resume_path)
            self._save_checkpoint(checkpoint['epoch'], save_latest=True)

        print("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))


def fix_module_in_state_dict(state_dict, model):
    load_state_dict_keys = list(state_dict.keys())
    curr_state_dict_keys = list(model.state_dict().keys())
    redo_dp = False
    if not curr_state_dict_keys[0].startswith('module.') and load_state_dict_keys[0].startswith('module.'):
        undo_dp = True
    elif curr_state_dict_keys[0].startswith('module.') and not load_state_dict_keys[0].startswith('module.'):
        redo_dp = True
        undo_dp = False
    else:
        undo_dp = False

    if undo_dp:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        # load params
    elif redo_dp:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = 'module.' + k  # remove `module.`
            new_state_dict[name] = v
    else:
        new_state_dict = state_dict
    return new_state_dict
