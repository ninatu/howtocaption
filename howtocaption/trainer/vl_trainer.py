import numpy as np
from torchvision import transforms
import os.path as op
import json
import pickle
from collections import defaultdict

from neptune.new.types import File
from pathlib import Path
import pandas as pd
from datetime import datetime

from howtocaption.base import BaseTrainer
from howtocaption.trainer.coco_eval import COCOEvalCap
from howtocaption.trainer.retrieval_eval import retrieval_metrics
from howtocaption.trainer.utils import _move_to_device
from howtocaption.utils.dist_utils import MetricLogger, is_dist_avail_and_initialized, get_rank, get_world_size, concat_all_gather, evenly_divisible_concat_all_gather, is_main_process
from torch.cuda.amp import autocast, GradScaler
from howtocaption.utils.retrieval_metrics_from_cap4video import compute_metrics, tensor_text_to_video_metrics, tensor_video_to_text_sim
from howtocaption.utils import inf_loop

from pycocotools.coco import COCO
import torch.nn.functional as F

import torch
import time
import os
import tqdm


def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad


class VL_Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, data_loader, valid_data_loader, lr_scheduler,
                 lr_scheduler_update='epoch',
                 save_latest=False,
                 mixed_precision=False,
                 log_step=50,
                 eval_args={},
                 use_longest_dataset_for_epoch_len=False,
                 len_epoch=None,
                 inf_dataloaders=False,
                 clip_grad=1e6,
                 accumulate_grad=False,
                 accumulate_grad_step=1,
                 **kwargs):

        self.clip_grad = clip_grad
        self.accumulate_grad = accumulate_grad
        self.accumulate_grad_step = accumulate_grad_step

        if not isinstance(data_loader, list):
            data_loader = [data_loader]
        if not isinstance(valid_data_loader, list):
            valid_data_loader = [valid_data_loader]

        self.train_data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_update = lr_scheduler_update
        self.save_latest = save_latest
        self.mixed_precision = mixed_precision
        self.log_step = log_step
        self.eval_args = eval_args

        self.use_longest_dataset_for_epoch_len = use_longest_dataset_for_epoch_len

        assert (len_epoch is None) or (use_longest_dataset_for_epoch_len is False)
        if len_epoch is None:
            assert inf_dataloaders is False

        self.inf_dataloaders = inf_dataloaders

        if len_epoch is not None:
            self.len_epoch = len_epoch
            if inf_dataloaders:
                self.train_data_loader = [inf_loop(d) for d in self.train_data_loader]
        elif use_longest_dataset_for_epoch_len:
            self.len_epoch = max([len(dl) for dl in self.train_data_loader])
        else:
            self.len_epoch = min([len(dl) for dl in self.train_data_loader])

        if self.mixed_precision:
            self.scaler = GradScaler()

        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print('Current Time', current_time, flush=True)

        super().__init__(**kwargs)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        start_epoch_time = time.time()

        torch.cuda.empty_cache()
        metric_logger = MetricLogger(delimiter="  ", neptune_run=self.neptune_run)

        # reshuffle dataset
        for dl in self.train_data_loader:
            if hasattr(dl, 'sampler') and dl.sampler is not None:
                dl.sampler.set_epoch(epoch)
            else:
                os.environ["WDS_EPOCH"] = str(epoch) # for webdataset

        self.model.train()

        batch_end_time = time.time()

        # cycle dataloader to the longest
        if self.inf_dataloaders:
            dataloaders = self.train_data_loader
        else:
            dataloaders = [(dl if len(dl) >= self.len_epoch else inf_loop(dl)) for dl in self.train_data_loader]
        iter_in_epoch = 0

        with tqdm.tqdm(zip(*dataloaders), desc=f"Training epoch {epoch}", total=self.len_epoch, miniters=10) as progress:
            for batch_idx, data_li in enumerate(progress):
                if iter_in_epoch >= self.len_epoch:
                    break
                iter_in_epoch += 1

                # measure data loading time
                metric_logger.update('data_time', time.time() - batch_end_time, step=self.step)

                # ------------------------------- optimize other losses -----------------------------------
                self.optimizer.zero_grad()

                for dl_idx, data in enumerate(data_li):

                    per_data_step = self.step * len(self.train_data_loader) + dl_idx
                    dataset_name = data['dataset'][0]
                    for key in ['video', 'image_embeds', 'text_embeds']:
                        if key in data:
                            data[key] = _move_to_device(data[key], self.device)

                    with autocast(enabled=self.mixed_precision):
                        output = self.model(data, n_queue=dl_idx)
                        loss = output['loss']

                        metric_logger.update(f'train/{dataset_name}/loss', loss.item(), step=self.step)
                        metric_logger.update('train/loss', loss.item(), step=per_data_step)

                        for key in output.keys():
                            if 'loss' in key:
                                metric_logger.update(f'train/{dataset_name}/{key}',
                                                     output[key].item(), step=per_data_step)

                        metric_logger.update(f'train/{dataset_name}/complete_loss', loss.item(), step=self.step)
                        metric_logger.update('train/complete_loss', loss.item(), step=per_data_step)

                    if self.mixed_precision:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    if not self.accumulate_grad:
                        if self.mixed_precision:
                            self.scaler.unscale_(self.optimizer)
                            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
                            self.optimizer.step()

                        self.optimizer.zero_grad()
                        metric_logger.update(f'train/{dataset_name}/grad', grad_norm.item(), step=self.step)
                        metric_logger.update(f'train/grad', grad_norm.item(), step=per_data_step)

                    if self.neptune_run is not None:
                        lr = self.optimizer.param_groups[0]["lr"]
                        self.neptune_run['lr'].log(lr, step=per_data_step)

                        # log for other parameter groups
                        for i in range(1, len(self.optimizer.param_groups)):
                            lr = self.optimizer.param_groups[i]["lr"]
                            self.neptune_run[f'lr_group{i}'].log(lr, step=per_data_step)

                    if batch_idx == 0:
                        # log input data
                        if (epoch % self.freq_visual_input == 0) or (self.log_visual_input_at_start and epoch == 1):
                            self._log_video(f'train/{dataset_name}/input', data['video'], epoch)
                        self._log_text(f'train/{dataset_name}/text', data['text'], epoch)
                    if batch_idx % self.log_step == 0:
                        print('\nTrain Epoch: {} {} Loss: {:.6f}'.format(
                            epoch,
                            self._progress(batch_idx),
                            metric_logger[f'train/{dataset_name}/loss'].avg), flush=True)

                if self.accumulate_grad and self.step % self.accumulate_grad_step == 0:
                    if self.mixed_precision:
                        self.scaler.unscale_(self.optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
                        self.optimizer.step()

                    self.optimizer.zero_grad()
                    metric_logger.update(f'train/{dataset_name}/grad', grad_norm.item(), step=self.step)
                    metric_logger.update(f'train/grad', grad_norm.item(), step=per_data_step)

                # apply lr scheduler
                if self.lr_scheduler_update == 'iter' and self.lr_scheduler is not None:
                    self.lr_scheduler.step(self.step)

                metric_logger.update('batch_time', time.time() - batch_end_time, step=self.step)
                batch_end_time = time.time()
                self.step += 1

                # ------------------------------- end of one iter -----------------------------------

        if is_dist_avail_and_initialized():
            torch.distributed.barrier()

        metric_logger.synchronize_between_processes()
        metric_logger.log_global_avg(epoch=epoch)
        log = metric_logger.get_global_avg()

        if self.save_latest:
            self._save_checkpoint(epoch, save_latest=True)

        metric_logger.update('train_epoch_time', time.time() - start_epoch_time, step=self.step)

        start_val_time = time.time()

        if epoch % self.retrieval_freq_eval == 0:
            self._eval_retrieval(epoch)

        metric_logger.update('retrieval val_epoch_time', time.time() - start_val_time, step=self.step)

        if epoch % self.nlp_freq_eval == 0:
            self._eval_nlp(epoch)

        if self.lr_scheduler_update == 'epoch' and self.lr_scheduler is not None:
            self.lr_scheduler.step(epoch)

        metric_logger.update('epoch_time', time.time() - start_epoch_time, step=self.step)
        return log

    def _log_video(self, name, video, epoch, max_n=6):
        if self.neptune_run is not None:
            video = video[:max_n]

            grid = torch.cat(list(video.transpose(1, 0)), dim=-1)
            grid = torch.cat(list(grid), dim=1)

            norm_mean = (0.485, 0.456, 0.406)
            norm_std = (0.229, 0.224, 0.225)
            grid = transforms.Normalize(mean=(0, 0, 0), std=1.0 / np.array(norm_std))(grid)
            grid = transforms.Normalize(mean=- np.array(norm_mean), std=(1, 1, 1))(grid)

            grid = grid.permute(1, 2, 0)
            grid = grid.cpu().numpy().clip(0, 1)

            grid = grid[:, :7168] # crop since neptune cannot large

            # log prediction
            self.neptune_run[name].log(File.as_image(grid), step=epoch)

    def _log_prediction(self, name, data, epoch, max_n=None, **kwargs):
        if self.neptune_run is not None and self.model_without_ddp.train_captioning:
            with torch.no_grad():  # important here, otherwise it hangs https://github.com/pytorch/pytorch/issues/54059
                train = self.model_without_ddp.training # save state
                self.model_without_ddp.eval()

                output_text = self.model_without_ddp(data, train=False, dont_update=True, **self.eval_args, **kwargs)
                output_text_sampled = self.model_without_ddp(data, train=False, dont_update=True,
                                                             sample=True,
                                                             **{k: v for k, v in self.eval_args.items() if k !='sample'},
                                                               **kwargs)

                self.neptune_run[name].log(f'Epoch {epoch}, step: {self.step}')
                if max_n is None:
                    max_n = len(data['text'])
                for i in range(max_n):
                    self.neptune_run[name].log(f'{i}    text: {data["text"][i]}')
                    self.neptune_run[name].log(f'{i}    pred: {output_text[i]}')
                    self.neptune_run[name].log(f'{i}    pred (sample): {output_text_sampled[i]}')

                if train:
                    self.model_without_ddp.train() # return state

    def _log_text(self, name, text, epoch, max_n=None):
        if self.neptune_run is not None:
            self.neptune_run[name].log(f'Epoch {epoch}, step: {self.step}')
            if max_n is None:
                max_n = len(text)
            for i in range(max_n):
                self.neptune_run[name].log(text[i])

    def _eval_nlp(self, epoch=None):
        if epoch is None:
            epoch = self.start_epoch
        available_datasets = ['MSRVTT_Cap', "MSVD", "YouCook2"]
        Path(self.config.save_dir).mkdir(parents=True, exist_ok=True)
        predictions_filename = f'{self.config.save_dir}/{self.config["name"]}_Epoch{epoch}_dataset_%s.json'
        if is_dist_avail_and_initialized():
            torch.distributed.barrier()

        print('Start NLP evaluation..', flush=True)
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print('Current Time', current_time, flush=True)

        world_size = get_world_size()
        if world_size == 1:
            cache_file = predictions_filename
        else:
            # local_rank would not work for cross-node distributed training
            cache_file = op.splitext(predictions_filename)[0] + '_{}_{}'.format(get_rank(),
                                                                                world_size) + \
                         op.splitext(predictions_filename)[1]

        train = self.model_without_ddp.training  # save state
        self.model_without_ddp.eval()

        with torch.no_grad():
            for _, dl in enumerate(self.valid_data_loader):
                if dl.dataset.dataset_name not in available_datasets:
                    continue

                res = []
                gt = []
                if 'MSRVTT' in dl.dataset.dataset_name:
                    json_fp = os.path.join(dl.dataset.data_root, 'annotation', 'MSR_VTT.json')
                    with open(json_fp, 'r') as fid:
                        data = json.load(fid)
                    df = pd.DataFrame(data['annotations'])
                elif dl.dataset.dataset_name == 'MSVD':
                    json_fp = os.path.join(dl.dataset.data_root, 'msvd_data', 'raw-captions.pkl')
                    with open(json_fp, 'rb') as fid:
                        df = pickle.load(fid)

                for i, data in enumerate(dl):
                    if i % 5 == 0:
                        print(f'Eval progress ({dl.dataset.dataset_name}): {i}/{len(dl)}')
                    data['video'] = _move_to_device(data['video'], self.device)
                    dataset_name = data['dataset'][0]

                    output_text = self.model_without_ddp(data, train=False,
                                                         dont_update=True, **self.eval_args)
                    for i in range(len(data['text'])):
                        if 'MSRVTT' in dl.dataset.dataset_name:
                            for cap_idx, gt_cap in enumerate(df[df['image_id'] == data['idx'][i]]['caption']):
                                gt.append({'caption': f'{gt_cap}', 'image_id': data['idx'][i], 'id': f"{data['idx'][i]}_{cap_idx}"})
                        elif dataset_name == 'MSVD':
                            for cap_idx, gt_cap in enumerate(df[data['idx'][i]]):
                                gt_cap = ' '.join(gt_cap)
                                gt.append({'caption': f'{gt_cap}', 'image_id': data['idx'][i], 'id': f"{data['idx'][i]}_{cap_idx}"})
                        elif dataset_name in ['DIDEMO']:
                            gt_cap = " ".join(dl.dataset.caption_dict[data['idx'][i]]['text'])
                            cap_idx = 0
                            gt.append({'caption': f'{gt_cap}', 'image_id': data['idx'][i], 'id': f"{data['idx'][i]}_{cap_idx}"})
                        elif dataset_name in ['LSMDC']:
                            gt_cap = dl.dataset.caption_dict[data['idx'][i]]['text'][0]
                            cap_idx = 0
                            gt.append({'caption': f'{gt_cap}', 'image_id': data['idx'][i], 'id': f"{data['idx'][i]}_{cap_idx}"})
                        elif dataset_name in ['YouCook2']:
                            gt_cap = dl.dataset.data[int(data['idx'][i])]['sentence']
                            cap_idx = 0
                            gt.append({'caption': f'{gt_cap}', 'image_id': data['idx'][i], 'id': f"{data['idx'][i]}_{cap_idx}"})

                        res.append({'caption': f'{output_text[i]}', 'image_id': data['idx'][i]})

                with open(cache_file % dataset_name, 'w') as f:
                    json.dump(res, f)
                    print(f'save: {cache_file % dataset_name}')

                with open(cache_file % f'{dataset_name}_gt', 'w') as f:
                    image_ids = set(i['image_id'] for i in gt)
                    image_ids = [{'id': image_id} for image_id in image_ids]
                    gt_save = {'annotations': gt, 'images': image_ids}
                    json.dump(gt_save, f)
                    print(f'save: {cache_file % f"{dataset_name}_gt"}')

                # collect all the files
                if world_size > 1:
                    torch.distributed.barrier()

                    if is_main_process():
                        # in the main stream
                        gt_annotations = []
                        gt_images = []
                        res_domain = []
                        for local_rank in range(world_size):
                            cur_cache_file = op.splitext(predictions_filename)[0] + '_{}_{}'.format(local_rank,
                                                                                                world_size) + \
                                         op.splitext(predictions_filename)[1]
                            with open(cur_cache_file % dataset_name, 'r') as f:
                                tmp = json.load(f)
                                res_domain.extend(tmp)

                            os.remove(cur_cache_file % dataset_name)

                            with open(cur_cache_file % f"{dataset_name}_gt", 'r') as f:
                                tmp = json.load(f)
                                gt_annotations.extend(tmp['annotations'])
                                gt_images.extend(tmp['images'])

                            os.remove(cur_cache_file % f"{dataset_name}_gt")

                        with open(predictions_filename % dataset_name, 'w') as f:
                            tmp_dict = defaultdict(int)
                            res_domain2 = []
                            for tmp_i in res_domain:
                                tmp_dict[tmp_i['image_id']] += 1
                                if tmp_dict[tmp_i['image_id']] == 1:
                                    res_domain2.append(tmp_i)
                            json.dump(res_domain2, f)
                            print(f'Sync and saved: {predictions_filename % dataset_name}')

                        gt_full = {
                            'annotations': gt_annotations,
                            'images': gt_images,
                        }

                        with open(predictions_filename % f"{dataset_name}_gt", 'w') as f:
                            json.dump(gt_full, f)
                            print(f'Sync and saved: {predictions_filename % f"{dataset_name}_gt"}', flush=True)

                if is_main_process():
                    coco = COCO(predictions_filename % f"{dataset_name}_gt")  # gt
                    coco_result = coco.loadRes(predictions_filename % dataset_name)
                    coco_eval = COCOEvalCap(coco, coco_result)
                    coco_eval.evaluate()

                    # print output evaluation scores
                    for metric, score in coco_eval.eval.items():
                        if self.neptune_run is not None:
                            self.neptune_run[f'{dataset_name}_{metric}'].log(score, step=epoch)
                        print(f'{metric}: {score:.3f}', flush=True)

                    metric_list = ["Bleu_4", "METEOR", "ROUGE_L", "CIDEr"]
                    print(f'{dataset_name} {metric_list}: ' + ' '.join([f"{coco_eval.eval[v] * 100:.1f}" for v in metric_list]), flush=True)

        if train:
            self.model_without_ddp.train()

    def _eval_retrieval(self, epoch=None):
        if epoch is None:
            epoch = self.start_epoch

        if not self.model_without_ddp.train_contrastive:
            return
        print('Eval retrieval', flush=True)
        # now = datetime.now()
        # current_time = now.strftime("%H:%M:%S")
        # print('Current Time', current_time, flush=True)
        train = self.model_without_ddp.training  # save state
        self.model_without_ddp.eval()

        with torch.no_grad():
            for _, dl in enumerate(self.valid_data_loader):

                # #################################################################
                ## below variables are used to multi-sentences retrieval
                # multi_sentence_: important tag for eval
                # cut_off_points: used to tag the label when calculate the metric
                # sentence_num: used to cut the sentence representation
                # video_num: used to cut the video representation
                # #################################################################
                multi_sentence_ = False
                cut_off_points_, sentence_num_, video_num_ = [], -1, -1
                if hasattr(dl.dataset, 'multi_sentence_per_video') \
                        and dl.dataset.multi_sentence_per_video:
                    multi_sentence_ = True
                    sentence_num_ = dl.dataset.sentence_num
                    video_num_ = len(dl.dataset.video_dict)
                    img_idxs = [-1 - get_rank()]

                if multi_sentence_:
                    print("Eval under the multi-sentence per video clip setting.")
                    print("sentence num: {}, video num: {}".format(sentence_num_, video_num_))

                image_feats = []
                text_feats = []

                total_sentence_number = 0
                for i, data in enumerate(dl):
                    data['video'] = _move_to_device(data['video'], self.device)
                    dataset_name = data['dataset'][0]

                    # get video features
                    image_embeds, _ = self.model_without_ddp._encode_image(data['video'], self.model_without_ddp.visual_encoder)
                    f = dl.dataset.num_frames
                    b, _, d = image_embeds.shape
                    image_embeds = image_embeds.view(b, f, -1, d)[:, :, 0, :].view(b * f, d)
                    image_feat = F.normalize(self.model_without_ddp.vision_proj(image_embeds).view(b, f, -1).mean(1), dim=-1)

                    # get text features
                    if multi_sentence_:
                        for within_batch_idx, video_idx in enumerate(data['idx']):
                            text_data = dl.dataset.get_full_caption(video_idx)

                            img_idxs.append(data['video_numerical_idx'][within_batch_idx])
                            max_text_length = data.get('max_text_length')[0].item()
                            text_embeds, _ = self.model_without_ddp._encode_text(text_data, self.model_without_ddp.text_encoder, max_text_length=max_text_length)
                            text_feat = F.normalize(self.model_without_ddp.text_proj(text_embeds[:, 0, :]), dim=-1)
                            text_feats.append(text_feat)

                            image_feats.append(image_feat[within_batch_idx].unsqueeze(0))
                            cut_off_points_.append(torch.tensor(text_feat.shape[0], device=text_feat.device, dtype=torch.float).view(1,-1))

                    else:
                        max_text_length = data.get('max_text_length')[0].item()
                        text_embeds, _ = self.model_without_ddp._encode_text(data['text'],self.model_without_ddp.text_encoder, max_text_length=max_text_length)
                        text_feat = F.normalize(self.model_without_ddp.text_proj(text_embeds[:, 0, :]), dim=-1)

                        text_feats.append(text_feat)
                        image_feats.append(image_feat)

                if multi_sentence_:
                    print('Total sentence number', total_sentence_number)

                    img_idxs = torch.tensor(img_idxs, device=image_feat.device)
                    img_idxs_all = concat_all_gather(img_idxs)
                    counter = {}
                    all_ranks = [-1 -i for i in range(get_world_size())]
                    to_del = []
                    for img_idx in img_idxs_all:
                        img_idx = img_idx.item()
                        if img_idx in all_ranks:
                            cur_rank = -1 - img_idx
                            img_enum = 0
                            continue

                        if img_idx not in counter:
                            counter[img_idx] = 1
                        else:
                            print(f'need to delete: {img_enum}, rank {get_rank()}, cur_rank {cur_rank}')
                            if cur_rank == get_rank():
                                to_del.append(img_enum)

                        img_enum += 1

                    for img_enum in sorted(to_del, reverse=True):
                        image_feats = image_feats[:img_enum] + image_feats[img_enum + 1:]
                        text_feats = text_feats[:img_enum] + text_feats[img_enum + 1:]
                        cut_off_points_ = cut_off_points_[:img_enum] + cut_off_points_[img_enum + 1:]
                    image_feats = torch.cat(image_feats, dim=0)
                    text_feats = torch.cat(text_feats, dim=0)
                    cut_off_points_ = torch.cat(cut_off_points_, dim=0)

                    image_feats = evenly_divisible_concat_all_gather(image_feats)
                    text_feats = evenly_divisible_concat_all_gather(text_feats)
                    cut_off_points_ = evenly_divisible_concat_all_gather(cut_off_points_)
                    print('Image feats', image_feats.shape, flush=True)
                    print('Text feats', text_feats.shape, flush=True)
                    print('Cut off points', cut_off_points_.shape, flush=True)
                else:
                    image_feats = torch.cat(image_feats, dim=0)
                    text_feats = torch.cat(text_feats, dim=0)
                    print('Total sentence number', total_sentence_number)

                    print('before gather image', image_feats.shape)
                    print('before gather text', text_feats.shape)
                    image_feats = concat_all_gather(image_feats)
                    text_feats = concat_all_gather(text_feats)
                    print('Image feats', image_feats.shape, flush=True)
                    print('Text feats', text_feats.shape, flush=True)

                sims_matrix = text_feats @ image_feats.t()

                if multi_sentence_:
                    sims_matrix = sims_matrix.cpu().detach().numpy()
                    cut_off_points_ = cut_off_points_.squeeze().detach().cpu().to(torch.int)
                    cut_off_points2len_ = cut_off_points_.cumsum(-1).tolist()
                    max_length = max([e_ - s_ for s_, e_ in zip([0] + cut_off_points2len_[:-1], cut_off_points2len_)])
                    sim_matrix_new = []
                    for s_, e_ in zip([0] + cut_off_points2len_[:-1], cut_off_points2len_):
                        sim_matrix_new.append(np.concatenate((sims_matrix[s_:e_],
                                                              np.full((max_length - e_ + s_, sims_matrix.shape[1]),
                                                                      -np.inf)), axis=0))
                    sim_matrix = np.stack(tuple(sim_matrix_new), axis=0)
                    tv_metrics = tensor_text_to_video_metrics(sim_matrix)
                    vt_metrics = compute_metrics(tensor_video_to_text_sim(sim_matrix))

                    ret_task = 't2v'
                    r1, r5, r10, r50 = tv_metrics["R1"], tv_metrics["R5"], tv_metrics["R10"], tv_metrics["R50"]
                    msg = f"[{ret_task}]{dataset_name:s} epoch {epoch}         {r1:.1f} {r5:.1f} {r10:.1f} {tv_metrics['MR']:g}"
                    msg += f"           {r50:.1f} {tv_metrics['MedianR']:g} {tv_metrics['MeanR']:.1f}"
                    print(msg)

                    ret_task = 'v2t'
                    r1, r5, r10, r50 = vt_metrics["R1"], vt_metrics["R5"], vt_metrics["R10"], vt_metrics["R50"]
                    msg = f"[{ret_task}]{dataset_name:s} epoch {epoch}         {r1:.1f} {r5:.1f} {r10:.1f} {vt_metrics['MR']:g}"
                    msg += f"           {r50:.1f} {vt_metrics['MedianR']:g} {vt_metrics['MeanR']:.1f}"
                    print(msg)

                    if self.neptune_run is not None:
                        ret_task = 't2v'
                        for metric, score in tv_metrics.items():
                            self.neptune_run[f'val/{dataset_name}/{ret_task}/{metric}'].log(score, step=epoch)

                        ret_task = 'v2t'
                        for metric, score in vt_metrics.items():
                            self.neptune_run[f'val/{dataset_name}/{ret_task}/{metric}'].log(score, step=epoch)

                else:
                    for ret_task, sims in [
                        ('t2v', sims_matrix),
                        ('v2t', sims_matrix.t())
                    ]:
                        metrics = retrieval_metrics(sims.cpu().numpy())

                        r1, r5, r10, r50 = metrics["R1"], metrics["R5"], metrics["R10"], metrics["R50"]
                        msg = f"[{ret_task}]{dataset_name:s} epoch {epoch} r1, r5, r10, MedR:    {r1:.1f} {r5:.1f} {r10:.1f} {metrics['MedR']:g}"
                        msg += f"    r50, MeanR:    {r50:.1f} {metrics['MeanR']:.1f}"
                        print(msg, flush=True)
                        if self.neptune_run is not None:
                            for metric, score in metrics.items():
                                self.neptune_run[f'val/{dataset_name}/{ret_task}/{metric}'].log(score, step=epoch)

        if train:
            self.model_without_ddp.train()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        current = batch_idx
        total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
