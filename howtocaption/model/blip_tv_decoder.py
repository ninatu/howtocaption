'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li

 Modified by Nina Shvetsova
'''
import copy

import torch
from torch import nn
import torch.nn.functional as F

from howtocaption.base import BaseModel
from howtocaption.model.utils.utils import init_tokenizer, load_checkpoint, create_vit_per_frame_model
from howtocaption.model.utils.utils import tie_encoder_decoder_weights as tie_encoder_decoder_weights_func
from howtocaption.model.utils.med import BertConfig, BertLMHeadModel, BertModel
from howtocaption.utils.dist_utils import concat_all_gather


class BlipVTDecoderModel(BaseModel):
    def __init__(self,
                 med_config='configs/med_config.json',
                 vit='base',
                 image_size=224,
                 queue_size=57600,
                 init_from_pretrained='https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth',
                 train_max_text_length=40,
                 tie_encoder_decoder_weights=True,
                 train_contrastive=False,
                 train_captioning=True,
                 train_itm=False,
                 continual_learning_weight=0,

                 # blip params
                 embed_dim=256,
                 momentum_encoder=True,
                 momentum=0.995,
                 alpha=0.4,
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """
        super().__init__()

        prompt = ''

        self.train_max_text_length = train_max_text_length
        self.visual_encoder, vision_width = create_vit_per_frame_model(image_size=image_size, vit=vit)
        self.tokenizer = init_tokenizer()
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width

        self.train_captioning = train_captioning
        self.train_itm = train_itm
        self.train_contrastive = train_contrastive
        self.momentum_encoder = momentum_encoder

        if train_captioning:
            self.text_decoder = BertLMHeadModel(config=med_config)
            self.text_decoder.resize_token_embeddings(len(self.tokenizer))

        if self.train_contrastive:
            text_encoder_config = copy.deepcopy(med_config)
            if train_captioning:
                text_encoder_config.add_cross_attention = True if tie_encoder_decoder_weights else False
            else:
                text_encoder_config.add_cross_attention = False

            self.text_encoder = BertModel(config=text_encoder_config, add_pooling_layer=False)
            self.text_encoder.resize_token_embeddings(len(self.tokenizer))
            text_width = self.text_encoder.config.hidden_size

            self.vision_proj = nn.Linear(vision_width, embed_dim)
            self.text_proj = nn.Linear(text_width, embed_dim)
            self.temp = nn.Parameter(0.07 * torch.ones([]))
            self.alpha = alpha

            if momentum_encoder:
                # create momentum encoders
                self.visual_encoder_m, vision_width = create_vit_per_frame_model(image_size=image_size, vit=vit)
                self.vision_proj_m = nn.Linear(vision_width, embed_dim)
                self.text_encoder_m = BertModel(config=text_encoder_config, add_pooling_layer=False)
                self.text_encoder_m.resize_token_embeddings(len(self.tokenizer))

                self.text_proj_m = nn.Linear(text_width, embed_dim)

                self.model_pairs = [[self.visual_encoder, self.visual_encoder_m],
                                    [self.vision_proj, self.vision_proj_m],
                                    [self.text_encoder, self.text_encoder_m],
                                    [self.text_proj, self.text_proj_m],
                                    ]
                # create the queue
                self.register_buffer("image_queue", torch.randn(embed_dim, queue_size))
                self.register_buffer("text_queue", torch.randn(embed_dim, queue_size))
                self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

                self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
                self.text_queue = nn.functional.normalize(self.text_queue, dim=0)

                self.queue_size = queue_size
                self.momentum = momentum
                self.copy_params()
        else:
            assert tie_encoder_decoder_weights == False
            self.text_encoder = None

        if self.train_itm:
            self.itm_head = nn.Linear(text_width, 2)

        # ---------  tie weights ----------------

        if tie_encoder_decoder_weights and train_captioning:
            print('tie_encoder_decoder_weights.....')
            skip_keys = ['/attention', 'word_embeddings'] # we don't tie word embeddings since otherwise initialization is not correct
            tie_encoder_decoder_weights_func(self.text_encoder, self.text_decoder.bert,'',skip_keys=skip_keys)

        # ---------  load pretrained BLIP weights ----------------

        if init_from_pretrained is not None:
            model, msg = load_checkpoint(self, init_from_pretrained)

            if queue_size != 57600:
                assert len(set(msg.missing_keys).difference(['image_queue', 'text_queue'])) == 0, msg.missing_keys
            else:
                assert (len(msg.missing_keys) == 0), msg.missing_keys

        self.prompt = prompt
        self.prompt_length = len(self.tokenizer(self.prompt).input_ids) - 1

        self.continual_learning_weight = continual_learning_weight
        if continual_learning_weight > 0:
            assert train_contrastive
            # create momentum encoders
            self.visual_encoder_original, vision_width = create_vit_per_frame_model(image_size=image_size, vit=vit)
            self.vision_proj_original = nn.Linear(vision_width, embed_dim)
            self.text_encoder_original = BertModel(config=text_encoder_config, add_pooling_layer=False)
            self.text_encoder_original.resize_token_embeddings(len(self.tokenizer))
            self.text_proj_original = nn.Linear(text_width, embed_dim)

            self.copy_params(
                model_pairs=[[self.visual_encoder, self.visual_encoder_original],
                              [self.vision_proj, self.vision_proj_original],
                              [self.text_encoder, self.text_encoder_original],
                              [self.text_proj, self.text_proj_original],
                              ]
            )

    def encode_image(self, video):
        image_embeds, _ =  self._encode_image(video, self.visual_encoder)

        b, f, _, _, _ = video.shape
        b, _, d = image_embeds.shape
        image_embeds = image_embeds.view(b, f, -1, d)[:, :, 0, :].view(b * f, d)

        image_feat = F.normalize(self.vision_proj(image_embeds), dim=-1)
        return image_feat

    def encode_text(self, text, use_momentum_encoder=False):
        if use_momentum_encoder:
            text_encoder = self.text_encoder_m
            text_proj = self.text_proj_m
        else:
            text_encoder = self.text_encoder
            text_proj = self.text_proj

        text_embeds, _ = self._encode_text(text, text_encoder)
        text_feat = F.normalize(text_proj(text_embeds[:, 0, :]), dim=-1)
        return text_feat

    def _encode_image(self, image, visual_encoder):
        image_embeds = visual_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        return image_embeds, image_atts

    def _encode_text(self, text, text_encoder, max_text_length=-1):
        if max_text_length == -1:
            max_text_length = self.train_max_text_length

        text = self.tokenizer(text, padding='longest', truncation=True, max_length=max_text_length,
                                  return_tensors="pt").to(text_encoder.device)
        text_output = text_encoder(text.input_ids, attention_mask=text.attention_mask,
                                  return_dict=True, mode='text')
        text_embeds = text_output.last_hidden_state
        return text_embeds, text.attention_mask

    def forward(self, data, train=True, dont_update=False, **kwargs):
        if train:
            return self.forward_train(data=data, dont_update=dont_update, **kwargs)
        else:
            return self.generate(data=data, **kwargs)

    def forward_train(self, data, dont_update=False, n_queue=0):
        device = data['video'].device if 'video' in data else data['image_embeds'].device
        image, caption = data['video'], data['text']

        #--------------- LM loss ---------------

        if self.train_captioning:
            image_embeds, image_atts = self._encode_image(data['video'], self.visual_encoder)
            text = self.tokenizer(caption, padding='longest', truncation=True, max_length=self.train_max_text_length,
                                          return_tensors="pt").to(device)

            text.input_ids[:, 0] = self.tokenizer.bos_token_id

            decoder_targets = text.input_ids.masked_fill(text.input_ids == self.tokenizer.pad_token_id, -100)
            decoder_targets[:, :self.prompt_length] = -100

            decoder_output = self.text_decoder(text.input_ids,
                                               attention_mask=text.attention_mask,
                                               encoder_hidden_states=image_embeds,
                                               encoder_attention_mask=image_atts,
                                               labels=decoder_targets,
                                               return_dict=True,
                                               )
            loss_lm = decoder_output.loss
        else:
            loss_lm = torch.tensor(0).to(device, non_blocking=True)

        #--------------- Contrastive loss ---------------

        if self.train_contrastive:
            b, f, _, _, _ = data['video'].shape
            if not self.train_captioning:
                image_embeds, image_atts = self._encode_image(data['video'], self.visual_encoder)
            b, _, d = image_embeds.shape

            image_feat = F.normalize(self.vision_proj(image_embeds.view(b, f, -1, d)[:, :, 0, :].view(b * f, d)).view(b, f, -1).mean(1), dim=-1)
            # get text features
            text_embeds, _ = self._encode_text(data['text'], self.text_encoder)
            text_embeds = text_embeds[:, 0, :]
            text_feat = F.normalize(self.text_proj(text_embeds), dim=-1)

            # get momentum features
            with torch.no_grad():
                if self.momentum_encoder:
                    if not dont_update:
                        self._momentum_update()
                    image_embeds_m, _ = self._encode_image(data['video'], self.visual_encoder_m)
                    text_embeds_m, _ = self._encode_text(data['text'], self.text_encoder_m)
                    b, f, _, _, _ = data['video'].shape
                    b, _, d = image_embeds_m.shape
                    image_embeds_m = image_embeds_m.view(b, f, -1, d)[:, :, 0, :].view(b * f, d)
                    image_feat_m = F.normalize(self.vision_proj_m(image_embeds_m).view(b, f, -1).mean(1), dim=-1)
                    image_feat_all = torch.cat([image_feat_m.t(), self.image_queue.clone().detach()], dim=1)

                    text_feat_m = F.normalize(self.text_proj_m(text_embeds_m[:, 0, :]), dim=-1)
                    text_feat_all = torch.cat([text_feat_m.t(), self.text_queue.clone().detach()], dim=1)

                    sim_i2t_m = image_feat_m @ text_feat_all / self.temp
                    sim_t2i_m = text_feat_m @ image_feat_all / self.temp

                else:
                    text_feat_all = text_feat.t()
                    image_feat_all = image_feat.t()

                    sim_i2t_m = image_feat @ text_feat_all / self.temp
                    sim_t2i_m = text_feat @ image_feat_all / self.temp

                sim_targets = torch.zeros(sim_i2t_m.size()).to(device)
                sim_targets.fill_diagonal_(1)

                sim_i2t_targets = (self.alpha * F.softmax(sim_i2t_m, dim=1) + (1 - self.alpha) * sim_targets).detach()
                sim_t2i_targets = (self.alpha * F.softmax(sim_t2i_m, dim=1) + (1 - self.alpha) * sim_targets).detach()

            sim_i2t = image_feat @ text_feat_all / self.temp
            sim_t2i = text_feat @ image_feat_all / self.temp

            loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1).mean()
            loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1).mean()

            ita_loss = (loss_i2t + loss_t2i) / 2

            if self.momentum_encoder and (not dont_update):
                self._dequeue_and_enqueue(image_feat_m, text_feat_m)

        else:
            ita_loss = torch.tensor(0).to(device, non_blocking=True)
            image_feat = None
            text_feat = None


        #--------------- Image-text Matching loss ---------------

        if self.train_itm:
            text = self.tokenizer(caption, padding='longest', truncation=True,
                                          max_length=self.train_max_text_length,
                                          return_tensors="pt").to(device)
            encoder_input_ids = text.input_ids.clone()
            encoder_input_ids[:, 0] = self.tokenizer.enc_token_id

            # forward the positve image-text pair
            bs = image.size(0)
            output_pos = self.text_encoder(encoder_input_ids,
                                           attention_mask=text.attention_mask,
                                           encoder_hidden_states=image_embeds,
                                           encoder_attention_mask=image_atts,
                                           return_dict=True,
                                           )

            with torch.no_grad():
                weights_t2i = F.softmax(sim_t2i[:, :bs], dim=1) + 1e-4
                weights_t2i.fill_diagonal_(0)
                weights_i2t = F.softmax(sim_i2t[:, :bs], dim=1) + 1e-4
                weights_i2t.fill_diagonal_(0)

            # select a negative image for each text
            image_embeds_neg = []
            for b in range(bs):
                neg_idx = torch.multinomial(weights_t2i[b], 1).item()
                image_embeds_neg.append(image_embeds[neg_idx])
            image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

            # select a negative text for each image
            text_ids_neg = []
            text_atts_neg = []
            for b in range(bs):
                neg_idx = torch.multinomial(weights_i2t[b], 1).item()
                text_ids_neg.append(encoder_input_ids[neg_idx])
                text_atts_neg.append(text.attention_mask[neg_idx])

            text_ids_neg = torch.stack(text_ids_neg, dim=0)
            text_atts_neg = torch.stack(text_atts_neg, dim=0)

            text_ids_all = torch.cat([encoder_input_ids, text_ids_neg], dim=0)
            text_atts_all = torch.cat([text.attention_mask, text_atts_neg], dim=0)

            image_embeds_all = torch.cat([image_embeds_neg, image_embeds], dim=0)
            image_atts_all = torch.cat([image_atts, image_atts], dim=0)

            output_neg = self.text_encoder(text_ids_all,
                                           attention_mask=text_atts_all,
                                           encoder_hidden_states=image_embeds_all,
                                           encoder_attention_mask=image_atts_all,
                                           return_dict=True,
                                           )

            vl_embeddings = torch.cat([output_pos.last_hidden_state[:, 0, :], output_neg.last_hidden_state[:, 0, :]], dim=0)
            vl_output = self.itm_head(vl_embeddings)

            itm_labels = torch.cat([torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
                                   dim=0).to(image.device)
            loss_itm = F.cross_entropy(vl_output, itm_labels)

        else:
            loss_itm =  torch.tensor(0).to(device, non_blocking=True)


        #--------------- Continual learning loss ---------------

        if self.continual_learning_weight > 0:
            with torch.no_grad():
                image_embeds_original, _ = self._encode_image(data['video'], self.visual_encoder_original)
                text_embeds_original, _ = self._encode_text(data['text'], self.text_encoder_original)
                b, f, _, _, _ = data['video'].shape
                b, _, d = image_embeds_original.shape
                image_embeds_original = image_embeds_original.view(b, f, -1, d)[:, :, 0, :].view(b * f, d)
                image_feat_original = F.normalize(self.vision_proj_original(image_embeds_original).view(b, f, -1).mean(1), dim=-1)
                text_feat_original = F.normalize(self.text_proj_original(text_embeds_original[:, 0, :]), dim=-1)

            continual_learning_loss = - (F.cosine_similarity(image_feat, image_feat_original).mean() +
                                         F.cosine_similarity(text_feat, text_feat_original).mean()) / 2
        else:
            continual_learning_loss = torch.tensor(0).to(device, non_blocking=True)

        loss = loss_lm + ita_loss + loss_itm + self.continual_learning_weight * continual_learning_loss

        return {'loss': loss,
                'lm_loss': loss_lm,
                'ita_loss': ita_loss,
                'itm_loss': loss_itm,
                'continual_learning_loss': continual_learning_loss,
                'image_embeds': image_embeds,
                }

    def generate(self, data, sample=False, num_beams=3, max_length=30, min_length=10, top_p=0.9,
                 repetition_penalty=1.0):
        image = data['video']

        image_embeds, image_atts = self._encode_image(data['video'], self.visual_encoder)

        if not sample:
            image_embeds = image_embeds.repeat_interleave(num_beams, dim=0)
            image_atts = image_atts.repeat_interleave(num_beams, dim=0)

        model_kwargs = {"encoder_hidden_states": image_embeds, "encoder_attention_mask": image_atts}

        prompt = [self.prompt] * image.size(0)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(image.device)

        input_ids[:, 0] = self.tokenizer.bos_token_id

        input_ids = input_ids[:, :-1]

        if sample:
            # nucleus sampling
            outputs = self.text_decoder.generate(input_ids=input_ids,
                                                 max_length=max_length,
                                                 min_length=min_length,
                                                 do_sample=True,
                                                 top_p=top_p,
                                                 num_return_sequences=1,
                                                 eos_token_id=self.tokenizer.sep_token_id,
                                                 pad_token_id=self.tokenizer.pad_token_id,
                                                 repetition_penalty=repetition_penalty,
                                                 **model_kwargs)
        else:
            # beam search
            outputs = self.text_decoder.generate(input_ids=input_ids,
                                                 max_length=max_length,
                                                 min_length=min_length,
                                                 num_beams=num_beams,
                                                 eos_token_id=self.tokenizer.sep_token_id,
                                                 pad_token_id=self.tokenizer.pad_token_id,
                                                 repetition_penalty=repetition_penalty,
                                                 **model_kwargs)

        captions = []
        for output in outputs:
            caption = self.tokenizer.decode(output, skip_special_tokens=True)
            captions.append(caption[len(self.prompt):])
        return captions

    @torch.no_grad()
    def copy_params(self, model_pairs=None):
        if model_pairs is None:
            model_pairs = self.model_pairs

        for model_pair in model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat):
        if self.queue_size == 0:
            return

        # gather keys before updating queue
        image_feats = concat_all_gather(image_feat)
        text_feats = concat_all_gather(text_feat)

        batch_size = image_feats.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr
