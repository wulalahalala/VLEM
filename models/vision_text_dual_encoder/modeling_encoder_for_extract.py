# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch VisionTextDualEncoder model."""
import sys
import os
sys.path.insert(0, '/data3/yiwei.ru/eye_movement_image')

from typing import Optional

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from transformers.modeling_utils import PreTrainedModel
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, \
    replace_return_docstrings
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.modeling_auto import AutoModel

from transformers.models.clip.modeling_clip import CLIPOutput, CLIPVisionConfig, CLIPVisionModel
from transformers.utils import ModelOutput

# my vit/swin
from models.vit.modeling_vit import ViTConfig, ViTModel
# from models.vit_subimage.modeling_vit import ViTConfig, ViTModel # sub-images
from models.swin.modeling_swin import SwinConfig, SwinModel

from .configuration_vision_text_dual_encoder import VisionTextDualEncoderForClassificationConfig

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import math
from scipy.stats import wasserstein_distance

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "VisionTextDualEncoderForClassificationConfig"

VISION_TEXT_DUAL_ENCODER_START_DOCSTRING = r"""
    This class can be used to initialize a vision-text dual encoder model with any pretrained vision autoencoding model
    as the vision encoder and any pretrained text model as the text encoder. The vision and text encoders are loaded
    via the [`~AutoModel.from_pretrained`] method. The projection layers are automatically added to the model and
    should be fine-tuned on a downstream task, like contrastive image-text modeling.
    In [LiT: Zero-Shot Transfer with Locked-image Text Tuning](https://arxiv.org/abs/2111.07991) it is shown how
    leveraging pre-trained (locked/frozen) image and text model for contrastive learning yields significant improvment
    on new zero-shot vision tasks such as image classification or retrieval.
    After such a Vision-Text-Dual-Encoder model has been trained/fine-tuned, it can be saved/loaded just like any other
    models (see the examples for more information).
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)
    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.
    Parameters:
        config ([`VisionEncoderDecoderConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

VISION_TEXT_DUAL_ENCODER_TEXT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.
            Indices can be obtained using [`PreTrainedTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.
            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            [What are attention masks?](../glossary#attention-mask)
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.
            [What are position IDs?](../glossary#position-ids)
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

VISION_TEXT_DUAL_ENCODER_VISION_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using
            [`CLIPFeatureExtractor`]. See [`CLIPFeatureExtractor.__call__`] for details.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

VISION_TEXT_DUAL_ENCODER_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.
            Indices can be obtained using [`CLIPTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.
            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            [What are attention masks?](../glossary#attention-mask)
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.
            [What are position IDs?](../glossary#position-ids)
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using
            a feature extractor (e.g. if you use ViT as the encoder, you should use [`ViTFeatureExtractor`]). See
            [`ViTFeatureExtractor.__call__`] for details.
        return_loss (`bool`, *optional*):
            Whether or not to return the contrastive loss.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@dataclass
class VisionTextClassifierOutput(ModelOutput):
    """
    Base class for outputs of sentence classification models.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    text_embeds: torch.FloatTensor = None
    image_embeds: torch.FloatTensor = None
    text_model_output: torch.FloatTensor = None
    vision_model_output: torch.FloatTensor = None


@add_start_docstrings(VISION_TEXT_DUAL_ENCODER_START_DOCSTRING)
class Attention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({hidden_size}) is not a multiple of the number of attention "
                f"heads ({num_attention_heads})"
            )

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(0.1)

    @add_start_docstrings_to_model_forward(VISION_TEXT_DUAL_ENCODER_INPUTS_DOCSTRING)
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    @add_start_docstrings_to_model_forward(VISION_TEXT_DUAL_ENCODER_INPUTS_DOCSTRING)
    def forward(
            self,
            hidden_states: torch.Tensor,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            mixed_query_layer = torch.unsqueeze(mixed_query_layer, dim=1)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        if is_cross_attention:
            new_attention_scores_shape = attention_scores.size()[:-1] + (14,-1)
            new_attention_scores = attention_scores.view(new_attention_scores_shape)
            attention_probs = nn.functional.softmax(new_attention_scores, dim=-2)
            attention_probs = attention_probs.permute(0, 1, 3, 4, 2)
        else:
            attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        if is_cross_attention:
            new_value_shpe = value_layer.size()[:2] +(14,14, value_layer.size()[-1])
            new_value = value_layer.view(new_value_shpe)
            time_fusion_patch = attention_probs*new_value
            context_layer = torch.sum(time_fusion_patch,dim=-2)
        else:
            context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


@add_start_docstrings(VISION_TEXT_DUAL_ENCODER_START_DOCSTRING)
class VisionTextDualEncoderModelForClassification(PreTrainedModel):
    config_class = VisionTextDualEncoderForClassificationConfig
    base_model_prefix = "vision_text_dual_encoder"

    def __init__(
            self,
            config: Optional[VisionTextDualEncoderForClassificationConfig] = None,
            vision_model: Optional[PreTrainedModel] = None,
            text_model: Optional[PreTrainedModel] = None,
            prediction_head_dropout: float = None
    ):

        if config is None and (vision_model is None or text_model is None):
            raise ValueError("Either a configuration or an vision and a text model has to be provided")

        if config is None:
            config = VisionTextDualEncoderForClassificationConfig.from_vision_text_configs(vision_model.config,
                                                                                           text_model.config)
        else:
            if not isinstance(config, self.config_class):
                raise ValueError(f"config: {config} has to be of type {self.config_class}")

        # initialize with config
        super().__init__(config)

        if vision_model is None:
            if isinstance(config.vision_config, CLIPVisionConfig):
                vision_model = CLIPVisionModel(config.vision_config)
            else:
                vision_model = AutoModel.from_config(config.vision_config)

        if text_model is None:
            text_model = AutoModel.from_config(config.text_config)

        self.vision_model = vision_model
        self.text_model = text_model

        # make sure that the individual model's config refers to the shared config
        # so that the updates to the config will be synced
        self.vision_model.config = self.config.vision_config
        self.text_model.config = self.config.text_config

        self.vision_embed_dim = config.vision_config.hidden_size
        self.text_embed_dim = config.text_config.hidden_size
        self.projection_dim = config.projection_dim
        self.num_classes = config.num_classes
        self.max_length = config.max_length
        self.patch_row_num = config.patch_row_num
        self.patch_column_num = config.patch_column_num
        self.cca_weight = config.cca_weight

        self.dropout = nn.Dropout(0.1)
        # Scanpath encoder
        self.gru = nn.GRU(input_size=768,
                          hidden_size=768,
                          num_layers=1,
                          batch_first=True,
                          bidirectional=False)

        self.num_attention_heads = 12
        # image embeds fusion layer
        
        self.image_fusion = nn.Linear(self.vision_embed_dim * self.patch_row_num, self.vision_embed_dim, bias=True)

        self.image_time_fusion = Attention(self.vision_embed_dim, self.num_attention_heads)
  
        self.image_gru =nn.GRU(input_size=768,
                          hidden_size=768,
                          num_layers=1,
                          batch_first=True,
                          bidirectional=False)
        
        self.image_fusion_activation = nn.GELU()

        #multimodel fusion layer
        # self attention layer
        self.fusion_embed_dim=self.vision_embed_dim+self.text_embed_dim
            #customer

        self.attention_head_size = int(self.fusion_embed_dim / self.num_attention_heads)
        self.wq = nn.Linear(self.fusion_embed_dim,self.fusion_embed_dim*self.num_attention_heads)
        self.wk = nn.Linear(self.fusion_embed_dim,self.fusion_embed_dim*self.num_attention_heads)
        self.wv = nn.Linear(self.fusion_embed_dim,self.fusion_embed_dim*self.num_attention_heads)
        self.softmax = nn.Softmax(dim=2)
        self.fc_o = nn.Linear(self.num_attention_heads * self.fusion_embed_dim, self.fusion_embed_dim)

            #paper
        '''
        self.fusion_attention_layer = Attention(self.fusion_embed_dim, self.num_attention_heads)
        '''
            #Ablation
        self.sqeuence_pooler=nn.Linear(15, 1, bias=False)

        # Classifier head
        if prediction_head_dropout is not None:
            self.prediction_head_dropout = nn.Dropout(prediction_head_dropout)
        else:
            self.prediction_head_dropout = nn.Dropout(0.1)

        self.projection = nn.Linear(self.vision_embed_dim+self.text_embed_dim, self.projection_dim, bias=False)
        self.activation = nn.Tanh()
        self.classifier = nn.Linear(self.projection_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()
        #PLM_AS
        self.plm_classifier = nn.Linear(768, self.num_classes)
        # loss
        self.layer_norm = nn.LayerNorm(15)
        self.image_project = nn.Linear(self.vision_embed_dim, 1)
        self.text_project = nn.Linear(self.text_embed_dim, 1)
        

    @add_start_docstrings_to_model_forward(VISION_TEXT_DUAL_ENCODER_TEXT_INPUTS_DOCSTRING)
    def get_text_features(
            self,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            token_type_ids=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        Returns:
            text_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The text embeddings obtained by
            applying the projection layer to the pooled output of [`CLIPTextModel`].
        Examples:
        ```python
        >>> from transformers import VisionTextDualEncoderModel, AutoTokenizer
        >>> model = VisionTextDualEncoderModel.from_pretrained("clip-italian/clip-italian")
        >>> tokenizer = AutoTokenizer.from_pretrained("clip-italian/clip-italian")
        >>> inputs = tokenizer(["una foto di un gatto", "una foto di un cane"], padding=True, return_tensors="pt")
        >>> text_features = model.get_text_features(**inputs)
        ```"""
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = text_outputs[1]

        return pooled_output

    @add_start_docstrings_to_model_forward(VISION_TEXT_DUAL_ENCODER_VISION_INPUTS_DOCSTRING)
    def get_image_features(
            self,
            pixel_values=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        Returns:
            image_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The image embeddings obtained by
            applying the projection layer to the pooled output of [`CLIPVisionModel`].
        Examples:
        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import VisionTextDualEncoderModel, AutoFeatureExtractor
        >>> model = VisionTextDualEncoderModel.from_pretrained("clip-italian/clip-italian")
        >>> feature_extractor = AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> inputs = feature_extractor(images=image, return_tensors="pt")
        >>> image_features = model.get_image_features(**inputs)
        ```"""
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = vision_outputs[1]  # pooled_output

        return pooled_output
    
    @add_start_docstrings_to_model_forward(VISION_TEXT_DUAL_ENCODER_VISION_INPUTS_DOCSTRING)
    def cca_loss( self,H1, H2):
        """
        It is the loss function of CCA as introduced in the original paper. There can be other formulations.
        """

        r1 = 1e-3
        r2 = 1e-3
        eps = 1e-9

        H1, H2 = H1.t(), H2.t()

        o1 =  H1.size(0)
        o2 = H2.size(0)

        m = H1.size(1)

        H1bar = H1 - H1.mean(dim=1).unsqueeze(dim=1)
        H2bar = H2 - H2.mean(dim=1).unsqueeze(dim=1)

 
        SigmaHat12 = (1.0 / (m - 1)) * torch.matmul(H1bar, H2bar.t())

        SigmaHat11 = (1.0 / (m - 1)) * torch.matmul(H1bar,
                                                    H1bar.t()) + r1 * torch.eye(o1, device=H1bar.device)
        SigmaHat22 = (1.0 / (m - 1)) * torch.matmul(H2bar,
                                                    H2bar.t()) + r2 * torch.eye(o2, device=H2bar.device)

        [D1, V1] = torch.symeig(SigmaHat11, eigenvectors=True)
        [D2, V2] = torch.symeig(SigmaHat22, eigenvectors=True)

        #modified
        D1 = D1.unsqueeze(1)    
        posInd1 = torch.gt(D1[:, 0], eps).nonzero()[:, 0]
        D1 = D1[posInd1, 0]
        V1 = V1[:, posInd1]
        D1 = torch.squeeze(D1)  # Remove extra dimensions from D1
        # Reshape D1 to 1-dimensional tensor
        D2 = D2.unsqueeze(1)
        posInd2 = torch.gt(D2, eps).nonzero()[:, 0]
        posInd2 = torch.gt(D2[:, 0], eps).nonzero()[:, 0]
        D2 = D2[posInd2, 0]
        V2 = V2[:, posInd2]
        D2 = torch.squeeze(D2)
        
        SigmaHat11RootInv = torch.matmul(
            torch.matmul(V1, torch.diag(D1 ** -0.5)), V1.t())
        SigmaHat22RootInv = torch.matmul(
            torch.matmul(V2, torch.diag(D2 ** -0.5)), V2.t())

        Tval = torch.matmul(torch.matmul(SigmaHat11RootInv,
                                         SigmaHat12), SigmaHat22RootInv)
  

        trace_TT = torch.matmul(Tval.t(), Tval)
        trace_TT = torch.add(trace_TT, (torch.eye(trace_TT.shape[0])*r1).to(trace_TT.device)) 
        U, V = torch.symeig(trace_TT, eigenvectors=True)
        U = torch.where(U>eps, U, (torch.ones(U.shape).float()*eps).to(trace_TT.device))
        U = U.topk(self.num_classes)[0]
        corr = torch.sum(torch.sqrt(U))
        return -corr

    
    @add_start_docstrings_to_model_forward(VISION_TEXT_DUAL_ENCODER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=VisionTextClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
            self,
            input_ids=None,
            pixel_values=None,
            attention_mask=None,
            position_ids=None,
            labels=None,
            token_type_ids=None,
            gaze_pos=None,
            patch_gaze_num=None,
            sample_id=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        Returns:
        Examples:
        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import (
        ...     VisionTextDualEncoderModel,
        ...     VisionTextDualEncoderProcessor,
        ...     ViTFeatureExtractor,
        ...     BertTokenizer,
        ... )
        >>> tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        >>> feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
        >>> processor = VisionTextDualEncoderProcessor(feature_extractor, tokenizer)
        >>> model = VisionTextDualEncoderModel.from_vision_text_pretrained(
        ...     "google/vit-base-patch16-224", "bert-base-uncased"
        ... )
        >>> # contrastive training
        >>> urls = [
        ...     "http://images.cocodataset.org/val2017/000000039769.jpg",
        ...     "https://farm3.staticflickr.com/2674/5850229113_4fe05d5265_z.jpg",
        ... ]
        >>> images = [Image.open(requests.get(url, stream=True).raw) for url in urls]
        >>> inputs = processor(
        ...     text=["a photo of a cat", "a photo of a dog"], images=images, return_tensors="pt", padding=True
        ... )
        >>> outputs = model(
        ...     input_ids=inputs.input_ids,
        ...     attention_mask=inputs.attention_mask,
        ...     pixel_values=inputs.pixel_values,
        ...     return_loss=True,
        ... )
        >>> loss, logits_per_image = outputs.loss, outputs.logits_per_image  # this is the image-text similarity score
        >>> # save and load from pretrained
        >>> model.save_pretrained("vit-bert")
        >>> model = VisionTextDualEncoderModel.from_pretrained("vit-bert")
        >>> # inference
        >>> outputs = model(**inputs)
        >>> logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        >>> probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
        ```"""

        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # text_outputs = self.text_model(
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     token_type_ids=token_type_ids,
        #     position_ids=position_ids,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        #     return_dict=return_dict
        # )

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=True,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return vision_outputs
        """Different models use different embeddings for classification.
        Swin and bert both use pooler_output (outputs[1]) for classification. 
        Vit and roberta both use outputs[0][:,0,:] for classification
        TODO: choose different embeddings for classification according to image and text models.
        """
        if self.vision_model.config.model_type == "swin":
            image_embeds = vision_outputs[1]
        elif self.vision_model.config.model_type == "vit":
            # image_embeds = vision_outputs[0][:,0,:]
            image_embeds = vision_outputs[0]
        else:
            raise Exception("Only support vit and swin vision model right now.")

        if self.text_model.config.model_type == "bert":
            text_embeds = text_outputs[0]
        elif self.text_model.config.model_type == "roberta":
            text_embeds = text_outputs[0][:, 0, :]
        else:
            raise Exception("Only support bert and roberta text model now.")

        # retrieve features according to scanpath ordering
        text_sp = torch.gather(text_embeds, 1, gaze_pos.unsqueeze(2).repeat(1, 1, 768).long())
        text_sp =self.dropout(text_sp)

        # encode text embeds according scanpath
        text_sp_len = (gaze_pos != self.max_length - 1).sum(1)
        text_sp_packed = pack_padded_sequence(text_sp, text_sp_len.cpu(), batch_first=True, enforce_sorted=False)
        gru_output_packed, gru_last_hidden = self.gru(text_sp_packed, text_embeds[:, 0, :].unsqueeze(0).contiguous())
        gru_output, gru_output_lengths = pad_packed_sequence(gru_output_packed, batch_first=True)
        gru_output = self.dropout(gru_output)
        
        #PLM_AS
        # gru_last_hidden=gru_last_hidden.squeeze(0)
        # logits = self.plm_classifier(gru_last_hidden)
        
        # retrieve text features according to patch
        # replace 0 with patch_max_gaze
        patch_max_gaze = torch.max(text_sp_len, 0)[0].item()
        replacement = torch.full(patch_gaze_num.size(), patch_max_gaze - 1).to(patch_gaze_num.device)
        patch_gaze_num_1 = torch.where(patch_gaze_num == -1, replacement, patch_gaze_num)
        patch_text_embeds = torch.gather(gru_output, 1, patch_gaze_num_1.unsqueeze(2).repeat(1, 1, 768).long())
        patch_text_embeds = torch.cat([text_embeds[:, 0, :].unsqueeze(1).contiguous(), patch_text_embeds], dim=1)
        
        
        # fuse image patch according to time
        
        image_patch_embeds = image_embeds[:, 1:, :]
        image_patch_embeds = image_patch_embeds.view(image_patch_embeds.shape[0], self.patch_column_num,self.patch_row_num, -1)
        image_patch_embeds = image_patch_embeds.permute(0, 2, 1, 3)
        time_patch_embeds = torch.flatten(image_patch_embeds, start_dim=2, end_dim=3)
        aggregate_image_embeds = self.image_fusion(time_patch_embeds)
        '''
        aggregate_image_embeds = self.image_time_fusion(image_embeds[:, 0, :],image_embeds[:, 1:, :])[0]
        '''
        aggregate_image_embeds = self.image_fusion_activation(aggregate_image_embeds)
        aggregate_image_embeds = self.dropout(aggregate_image_embeds)
        patch_image_embeds,gru_last_hidden = self.image_gru(aggregate_image_embeds,image_embeds[:, 0, :].unsqueeze(0).contiguous())
        patch_image_embeds = self.dropout(patch_image_embeds)
        patch_image_embeds = torch.cat([image_embeds[:, 0, :].unsqueeze(1).contiguous(), patch_image_embeds], dim=1)
        # patch_image_embeds = torch.cat([image_embeds[:, 0, :].unsqueeze(1).contiguous(), aggregate_image_embeds], dim=1)
        

        # fuse image and text embeds
        patch_image_embeds = self.layer_norm(patch_image_embeds.transpose(1,2)).transpose(1,2)
        patch_text_embeds = self.layer_norm(patch_text_embeds.transpose(1,2)).transpose(1,2)

        out_image_embeds = patch_image_embeds.detach().cpu().numpy()
        out_text_embeds = patch_text_embeds.detach().cpu().numpy()
        out_labels = labels.detach().cpu().numpy()
        image_path = './features/imageWOCCAWD.npy'
        text_path = './features/textWOCCAWD.npy'
        label_path = './features/label.npy'
        if os.path.exists(image_path):
            all_image = np.load(image_path)
            all_text = np.load(text_path)
            all_label = np.load(label_path)
            out_image_embeds = np.concatenate((all_image,out_image_embeds))
            out_text_embeds = np.concatenate((all_text,out_text_embeds))
            out_labels = np.concatenate((all_label,out_labels))

        np.save(image_path,out_image_embeds)
        np.save(text_path,out_text_embeds)
        np.save(label_path,out_labels)
        
        vl_embeds = torch.cat([patch_image_embeds, patch_text_embeds], dim=-1)
        '''
        pooled_output = self.sqeuence_pooler(vl_embeds.transpose(1,2)).squeeze()
        pooled_output = self.activation(pooled_output)
        pooled_output = self.dropout(pooled_output)
        pooled_output = self.projection(pooled_output)
        '''
        q = self.wq(vl_embeds)
        k = self.wk(vl_embeds)
        v = self.wv(vl_embeds)
        q = q.view(vl_embeds.shape[0], vl_embeds.shape[1], self.num_attention_heads, self.fusion_embed_dim).permute(2, 0, 1, 3).contiguous().view(-1, vl_embeds.shape[1], self.fusion_embed_dim)
        k = k.view(vl_embeds.shape[0], vl_embeds.shape[1], self.num_attention_heads, self.fusion_embed_dim).permute(2, 0, 1, 3).contiguous().view(-1, vl_embeds.shape[1], self.fusion_embed_dim)
        v = v.view(vl_embeds.shape[0], vl_embeds.shape[1], self.num_attention_heads, self.fusion_embed_dim).permute(2, 0, 1, 3).contiguous().view(-1, vl_embeds.shape[1], self.fusion_embed_dim)
        u = torch.bmm(q, k.transpose(1, 2))
        u = u / np.power(self.fusion_embed_dim, 0.5)
        attn = self.softmax(u)
        attn = self.dropout(attn)
        fusion_embeds = torch.bmm(attn, v)
        fusion_embeds = fusion_embeds.view(self.num_attention_heads, vl_embeds.shape[0], vl_embeds.shape[1], self.fusion_embed_dim).permute(1, 2, 0, 3).contiguous().view(vl_embeds.shape[0], vl_embeds.shape[1], -1)
        fusion_embeds = self.fc_o(fusion_embeds)
        
        pooled_output = self.projection(fusion_embeds[:, 0, :])
        
        pooled_output = self.activation(pooled_output)
        pooled_output = self.prediction_head_dropout(pooled_output)


        # classification logits
        logits = self.classifier(pooled_output)

        # #CCA WD loss
        projected_text = self.text_project(patch_text_embeds)
        projected_image = self.image_project(patch_image_embeds)
        projected_text = torch.squeeze(projected_text, dim=2)
        projected_image = torch.squeeze(projected_image, dim=2)

        # calculate classification loss
        loss = None
        if labels is not None:
            if self.num_classes == 1:
                self.config.problem_type = "regression"
            elif self.num_classes > 1 and (labels.dtype == torch.long or labels.dtype == torch.int) and labels.shape[
                1] != 11:
                self.config.problem_type = "single_label_classification"
            else:
                self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_classes == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)


        loss = loss+self.cca_weight*self.cca_loss(projected_text, projected_image)
        loss = loss + torch.tensor(wasserstein_distance(projected_text.cpu().detach().numpy().flatten(), projected_image.cpu().detach().numpy().flatten()), requires_grad=True)

        if not return_dict:
            output = (logits)
            return ((loss,) + output) if loss is not None else output

        return VisionTextClassifierOutput(
            loss=loss,
            logits=logits
        )


    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        # At the moment fast initialization is not supported
        # for composite models
        kwargs["_fast_init"] = False
        return super().from_pretrained(*args, **kwargs)

    @classmethod
    def from_vision_text_pretrained(
            cls,
            vision_model_name_or_path: str = None,
            text_model_name_or_path: str = None,
            num_classes: int = None,
            max_length: int = None,
            *model_args,
            **kwargs,
    ) -> PreTrainedModel:
        """
        Params:
            vision_model_name_or_path (`str`, *optional*, defaults to `None`):
                Information necessary to initiate the vision model. Can be either:
                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                      Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a
                      user or organization name, like `dbmdz/bert-base-german-cased`.
                    - A path to a *directory* containing model weights saved using
                      [`~FlaxPreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.
                    - A path or url to a *PyTorch checkpoint folder* (e.g, `./pt_model`). In this case, `from_pt`
                      should be set to `True` and a configuration object should be provided as `config` argument. This
                      loading path is slower than converting the PyTorch checkpoint in a Flax model using the provided
                      conversion scripts and loading the Flax model afterwards.
            text_model_name_or_path (`str`, *optional*):
                Information necessary to initiate the text model. Can be either:
                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                      Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a
                      user or organization name, like `dbmdz/bert-base-german-cased`.
                    - A path to a *directory* containing model weights saved using
                      [`~FlaxPreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.
                    - A path or url to a *PyTorch checkpoint folder* (e.g, `./pt_model`). In this case, `from_pt`
                      should be set to `True` and a configuration object should be provided as `config` argument. This
                      loading path is slower than converting the PyTorch checkpoint in a Flax model using the provided
                      conversion scripts and loading the Flax model afterwards.
            model_args (remaining positional arguments, *optional*):
                All remaning positional arguments will be passed to the underlying model's `__init__` method.
            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
                `output_attentions=True`).
                - To update the text configuration, use the prefix *text_* for each configuration parameter.
                - To update the vision configuration, use the prefix *vision_* for each configuration parameter.
                - To update the parent model configuration, do not use a prefix for each configuration parameter.
                Behaves differently depending on whether a `config` is provided or automatically loaded.
        Example:
        ```python
        >>> from transformers import VisionTextDualEncoderModel
        >>> # initialize a model from pretrained ViT and BERT models. Note that the projection layers will be randomly initialized.
        >>> model = VisionTextDualEncoderModel.from_vision_text_pretrained(
        ...     "google/vit-base-patch16-224", "bert-base-uncased"
        ... )
        >>> # saving model after fine-tuning
        >>> model.save_pretrained("./vit-bert")
        >>> # load fine-tuned model
        >>> model = VisionTextDualEncoderModel.from_pretrained("./vit-bert")
        ```"""
        kwargs_vision = {
            argument[len("vision_"):]: value for argument, value in kwargs.items() if argument.startswith("vision_")
        }

        kwargs_text = {
            argument[len("text_"):]: value for argument, value in kwargs.items() if argument.startswith("text_")
        }

        # remove vision, text kwargs from kwargs
        for key in kwargs_vision.keys():
            del kwargs["vision_" + key]
        for key in kwargs_text.keys():
            del kwargs["text_" + key]

        # Load and initialize the vision and text model
        vision_model = kwargs_vision.pop("model", None)
        if vision_model is None:
            if vision_model_name_or_path is None:
                raise ValueError(
                    "If `vision_model` is not defined as an argument, a `vision_model_name_or_path` has to be defined"
                )

            if "config" not in kwargs_vision:
                vision_config = AutoConfig.from_pretrained(vision_model_name_or_path)

            if vision_config.model_type == "clip":
                kwargs_vision["config"] = vision_config.vision_config
                vision_model = CLIPVisionModel.from_pretrained(vision_model_name_or_path, *model_args, **kwargs_vision)
                # TODO: Should we use the pre-trained projection as well ?
            elif vision_config.model_type == "vit":
                # our modified vit
                vision_model = ViTModel.from_pretrained(vision_model_name_or_path, *model_args, **kwargs_vision)
            elif vision_config.model_type == "swin":
                # our modified swin
                vision_model = SwinModel.from_pretrained(vision_model_name_or_path, *model_args, **kwargs_vision)
            else:
                kwargs_vision["config"] = vision_config
                vision_model = AutoModel.from_pretrained(vision_model_name_or_path, *model_args, **kwargs_vision)

        text_model = kwargs_text.pop("model", None)
        if text_model is None:
            if text_model_name_or_path is None:
                raise ValueError(
                    "If `text_model` is not defined as an argument, a `text_model_name_or_path` has to be defined"
                )

            if "config" not in kwargs_text:
                text_config = AutoConfig.from_pretrained(text_model_name_or_path)
                kwargs_text["config"] = text_config

            text_model = AutoModel.from_pretrained(text_model_name_or_path, *model_args, **kwargs_text)

        # instantiate config with corresponding kwargs
        config = VisionTextDualEncoderForClassificationConfig.from_vision_text_configs(vision_model.config,
                                                                                       text_model.config, num_classes,
                                                                                       max_length, **kwargs)

        # init model
        model = cls(config=config, vision_model=vision_model, text_model=text_model)

        # the projection layers are always newly initialized when loading the model
        # using pre-trained vision and text model.
        logger.warning(
            "The projection layer and logit scale weights `['visual_projection.weight', 'text_projection.weight',"
            " 'logit_scale']` are newly initialized. You should probably TRAIN this model on a down-stream task to be"
            " able to use it for predictions and inference."
        )

        return model