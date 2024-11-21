import pandas as pd
import copy
import transformers
from transformers import Trainer
from transformers import TrainerState
from transformers import TrainerControl
from transformers import TrainingArguments
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
import pymongo as pm
from tvts.tvts import Tvts


def check_label_ids(xdataset, xproperty):
    df = xdataset.to_pandas()
    label_np = df[xproperty].to_numpy()
    ids = []
    for inner_arr in label_np:
        for id in inner_arr:
            ids.append(id)
    ser = pd.Series(ids)
    counts = ser.value_counts()
    return counts


class LogCallback(transformers.TrainerCallback):
    def __init__(
        self, ts, logger, 
        steps_when_eval, steps_when_save, 
        save_path, 
        epoch_base_hf=0, epoch_base_tvts=0.0, is_batch_global=True,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.ts: Tvts = ts
        self.logger = logger
        self.steps_when_eval = steps_when_eval
        self.eval_steps = steps_when_eval
        self.steps_when_save = steps_when_save
        self.save_steps = steps_when_save
        self.save_path = save_path
        self.epoch_base_hf = epoch_base_hf
        self.epoch_base_tvts = epoch_base_tvts
        self.is_batch_global = is_batch_global
        # self.loss = 0.
        # self.n_batch = 0
        self.first = True

    """
    https://discuss.huggingface.co/t/logs-of-training-and-validation-loss/1974/3
    """

    def on_train_begin(self, args, state, control, **kwargs):
        if args.process_index:
            self.logger.debug(f'**** Return in process #{args.process_index}')
            return
        self.logger.debug(f'on train begin: state.eval_steps {state.eval_steps}=>{self.eval_steps} state.save_steps {state.save_steps}=>{self.save_steps}')
        assert state.eval_steps == self.eval_steps
        assert state.save_steps == self.save_steps

    def on_step_end(self, args, state, control, **kwargs):
        """
        https://discuss.huggingface.co/t/logs-of-training-and-validation-loss/1974/6
        """
        if args.process_index:
            self.logger.debug(f'**** Return in process #{args.process_index}')
            return
        self.logger.debug('on_step_end step: %d, len of log_history: %d', state.global_step, len(state.log_history))
        if state.global_step == 1:

            return
        if (state.global_step - 1) % self.steps_when_eval == 0:
            history = state.log_history[-2]
        else:
            history = state.log_history[-1]
            # self.loss += history['loss']
            # self.n_batch += 1
        self.log_step(state.global_step - 1, history)

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if args.process_index:
            self.logger.debug(f'**** Return in process #{args.process_index}')
            return
        self.logger.debug('on_epoch_end step: %d, len of log_history: %d', state.global_step, len(state.log_history))

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if args.process_index:
            self.logger.debug(f'**** Return in process #{args.process_index}')
            return
        self.logger.debug('on_evaluate step: %d, len of log_history: %d', state.global_step, len(state.log_history))
        # self.loss += state.log_history[-2]['loss']
        # self.n_batch += 1
        # loss = self.loss / self.n_batch
        # self.loss = 0.
        # self.n_batch = 0
        self.log_epoch(state.global_step, state.log_history[-2], state.log_history[-1])

    def on_train_end(self, args, state, control, **kwargs):
        if args.process_index:
            self.logger.debug(f'**** Return in process #{args.process_index}')
            return
        self.logger.debug('on_train_end step: %d, len of log_history: %d', state.global_step, len(state.log_history))
        history = None
        for history in state.log_history[::-1]:
            loss = history.get('loss', None)
            if loss is not None:
                self.log_step(state.global_step, history)
                break

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if args.process_index:
            self.logger.debug(f'**** Return in process #{args.process_index}')
            return
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        self.logger.debug('on_save step: %d, len of log_history: %d, checkpoint_folder: %s', state.global_step, len(state.log_history), checkpoint_folder)
        history = state.log_history[-1]
        if history.get('eval_loss', None) is None:
            self.logger.debug('on_save step: %d, len of log_history: %d, checkpoint_folder: %s: DO NOT log in tvts because of no eval_loss', state.global_step, len(state.log_history), checkpoint_folder)
        else:
            self.log_save(state.global_step, history, checkpoint_folder, self.save_path)
        
    def translate_epoch(self, epoch):
        epoch = epoch + self.epoch_base_tvts - self.epoch_base_hf
        epoch = round(epoch, 2)
        return epoch
        
        
    def log_step(self, xstep, xdict):
        xdict = copy.deepcopy(xdict)
        epoch = self.translate_epoch(xdict['epoch'])
        del xdict['epoch']
        xdict['the_epoch'] = epoch
        xdict['step'] = xstep
        self.ts.save_batch(epoch, xstep, xdict, is_batch_global=self.is_batch_global)


    def log_epoch(self, xstep, xdict, xdict_train):
        xdict = copy.deepcopy(xdict)
        xdict_train = copy.deepcopy(xdict_train)
        xdict.update(**xdict_train)
        epoch = self.translate_epoch(xdict['epoch'])
        del xdict['epoch']
        xdict['the_epoch'] = epoch
        xdict['step'] = xstep
        
        # loss in this epoch
        # Warning: due to hard wound of Huggingface callbacks and state.log_history, the last batch's loss cannot be accumulated.
        cur = self.ts.table_4batch.find({
            'train_id': self.ts.train_id,
        }).sort([
            ('epoch', pm.ASCENDING),
            ('batch', pm.ASCENDING)
        ])
        loss = 0.
        n = 0
        for d in cur[xstep - self.eval_steps:xstep]:
            n += 1
            loss += d['loss']
        loss /= n
        
        xdict['loss'] = loss
        self.ts.save_epoch(epoch, xdict)


    def log_save(self, xstep, xdict, rel_path, xdir):
        xdict = copy.deepcopy(xdict)
        epoch = self.translate_epoch(xdict['epoch'])
        del xdict['epoch']
        xdict['the_epoch'] = epoch
        xdict['step'] = xstep
        self.ts.save_epoch(epoch, xdict, save_rel_path=rel_path, save_dir=xdir)
