"""
modeling_prismatic.py

Core HuggingFace-style PrismaticPreTrainedModel and PrismaticForConditionalGeneration class definitions, inheriting
from the default `transformers.PretrainedModel`. Meant to be standalone and self-contained, but exactly replicate the
logic in `prismatic.models.vlms.prismatic.py`.

Note =>> for the time being, not adding the custom HF "docstring" formatting.

References [LLaVa, IDEFICS-2]:
    => https://github.com/huggingface/transformers/blob/main/src/transformers/models/llava/modeling_llava.py
    => https://github.com/huggingface/transformers/blob/main/src/transformers/models/idefics2/modeling_idefics2.py
"""

import logging
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple, Union

import numpy as np
import timm
import tokenizers
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from timm.models.vision_transformer import LayerScale
from transformers import AutoModelForCausalLM, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import ModelOutput
import GPUtil

from .configuration_prismatic import OpenVLAConfig, PrismaticConfig, MoEOpenVLAConfig
from .layers.moe import ActionMoELayer

# Get Logger
logger = logging.getLogger(__name__)


# === PyTorch/HuggingFace Default IGNORE_INDEX (for CrossEntropyLoss labels)
IGNORE_INDEX = -100


# === Utility Functions for Monkey-Patching ===
def unpack_tuple(fn: Callable[[Any], Tuple[Any]]) -> Callable[[Any], Any]:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = fn(*args, **kwargs)
        return result[0] if isinstance(result, tuple) else result

    return wrapper


# HF Transformers overwrites parameters with names containing `gamma`; we're going to patch VisionBackbone.LayerScale.
#   =>> TIMM :: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py#L109
#   =>> Transformers :: https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L3960
def _ls_new_forward(self, x: torch.Tensor) -> torch.Tensor:
    return x.mul_(self.scale_factor) if self.inplace else x * self.scale_factor


def ls_apply_patch(ls_module: LayerScale):
    ls_module.scale_factor = nn.Parameter(ls_module.gamma.clone())
    ls_module.forward = _ls_new_forward.__get__(ls_module, LayerScale)
    del ls_module.gamma


# === Prismatic Vision Backbone (nn.Module) Definitions (w/ Fused Backbone Support) ===
class PrismaticVisionBackbone(nn.Module):
    def __init__(
        self,
        use_fused_vision_backbone: bool,
        image_sizes: List[int],
        timm_model_ids: List[str],
        timm_override_act_layers: List[Optional[str]],
    ) -> None:
        super().__init__()
        self.use_fused_vision_backbone = use_fused_vision_backbone

        # [Contract] Validate number of (fused) vision backbones, create "alpha" featurizer and Instantiate
        #   =>> Note :: Monkey-Patch the `forward()` function of the backbone to ensure FSDP-compatibility
        #               Hardcodes `get_intermediate_layers` to return the **SECOND-TO-LAST** layer patches!
        assert len(timm_model_ids) <= 2, "Prismatic models only support up to 2 (fused) vision backbones!"
        self.featurizer = timm.create_model(
            timm_model_ids[0],
            pretrained=False,
            num_classes=0,
            img_size=image_sizes[0],
            act_layer=timm_override_act_layers[0],
        )
        self.featurizer.forward = unpack_tuple(
            partial(self.featurizer.get_intermediate_layers, n={len(self.featurizer.blocks) - 2})
        )
        self.embed_dim = self.featurizer.embed_dim

        # If `use_fused_vision_backbone` =>> create "beta" featurizer
        if self.use_fused_vision_backbone:
            self.fused_featurizer = timm.create_model(
                timm_model_ids[1],
                pretrained=False,
                num_classes=0,
                img_size=image_sizes[1],
                act_layer=timm_override_act_layers[1],
            )
            self.fused_featurizer.forward = unpack_tuple(
                partial(self.fused_featurizer.get_intermediate_layers, n={len(self.fused_featurizer.blocks) - 2})
            )
            self.embed_dim += self.fused_featurizer.embed_dim

        # Patch `vision_backbone.featurizer` and `vision_backbone.fused_featurizer` with HF-Compatible LayerScale
        for module in self.featurizer.modules():
            if isinstance(module, LayerScale):
                ls_apply_patch(module)

        if self.use_fused_vision_backbone:
            for module in self.fused_featurizer.modules():
                if isinstance(module, LayerScale):
                    ls_apply_patch(module)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run image (`pixel_values`) through featurizer; if channel-stacked, then dispatch and sequence stack."""
        if not self.use_fused_vision_backbone:
            return self.featurizer(pixel_values)

        # Split `pixel_values :: [bsz, 2 * 3, resolution, resolution]` =>> featurize =>> channel stack
        img, img_fused = torch.split(pixel_values, [3, 3], dim=1)
        patches, patches_fused = self.featurizer(img), self.fused_featurizer(img_fused)

        return torch.cat([patches, patches_fused], dim=2)


# === Prismatic Projector (nn.Module) Definitions ===
class PrismaticProjector(nn.Module):
    def __init__(self, use_fused_vision_backbone: bool, vision_dim: int, llm_dim: int) -> None:
        super().__init__()
        self.use_fused_vision_backbone = use_fused_vision_backbone
        self.vision_dim, self.llm_dim = vision_dim, llm_dim

        # Switch on `use_fused_vision_backbone` =>> use slightly different MLPs and projection factors!
        if not self.use_fused_vision_backbone:
            self.fc1 = nn.Linear(self.vision_dim, self.llm_dim, bias=True)
            self.fc2 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
            self.act_fn1 = nn.GELU()
        else:
            initial_projection_dim = 4 * vision_dim
            self.fc1 = nn.Linear(self.vision_dim, initial_projection_dim, bias=True)
            self.fc2 = nn.Linear(initial_projection_dim, self.llm_dim, bias=True)
            self.fc3 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
            self.act_fn1 = nn.GELU()
            self.act_fn2 = nn.GELU()

    def forward(self, img_patches: torch.Tensor) -> torch.Tensor:
        if not self.use_fused_vision_backbone:
            projected_features = self.fc1(img_patches)
            projected_features = self.act_fn1(projected_features)
            projected_features = self.fc2(projected_features)
        else:
            projected_features = self.fc1(img_patches)
            projected_features = self.act_fn1(projected_features)
            projected_features = self.fc2(projected_features)
            projected_features = self.act_fn2(projected_features)
            projected_features = self.fc3(projected_features)

        return projected_features


# === Main HF Class Definitions ===
@dataclass
class PrismaticCausalLMOutputWithPast(ModelOutput):
    """Base class for Prismatic casual (visually-conditioned) language model outputs; also exposes visual features."""

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

    # Additions for VLMs
    projector_features: Optional[torch.FloatTensor] = None


class PrismaticPreTrainedModel(PreTrainedModel):
    config_class: PretrainedConfig = PrismaticConfig
    base_model_prefix: str = "model"
    supports_gradient_checkpointing: bool = True

    _no_split_modules: ClassVar[List[str]] = ["PrismaticProjector"]
    _skip_keys_device_placement: str = "past_key_values"
    _supports_flash_attn_2: bool = True

    def _init_weights(self, module: nn.Module) -> None:
        # Important :: this HF ported version is *not* meant for training from scratch; only inference and fine-tuning!
        #   => As such, this init_weights code is not correct; if training VLMs from scratch, use the main codebase at
        #      https://github.com/TRI-ML/prismatic-vlms
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.text_config.initializer_range
        )

        if hasattr(module, "class_embedding"):
            module.class_embedding.data.normal_(mean=0.0, std=std)

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @property
    def _supports_sdpa(self) -> bool:
        """Check LLM supports SDPA Attention"""
        return self.language_model._supports_sdpa


class PrismaticForConditionalGeneration(PrismaticPreTrainedModel):
    def __init__(self, config: PrismaticConfig) -> None:
        super().__init__(config)

        # [Validation] Lightweight Validate on `config` Fields + Dependency Versions
        if config.use_fused_vision_backbone is None:
            raise ValueError("Missing config field `use_fused_vision_backbone`")

        if timm.__version__ not in {"0.9.10", "0.9.11", "0.9.12", "0.9.16"}:
            raise NotImplementedError(
                "TIMM Version must be >= 0.9.10 and < 1.0.0 (breaking); please raise a GitHub Issue "
                "if you urgently need support for latest TIMM versions."
            )

        if (transformers.__version__ != "4.40.1") or (tokenizers.__version__ != "0.19.1"):
            logger.warning(
                f"Expected `transformers==4.40.1` and `tokenizers==0.19.1` but got "
                f"`transformers=={transformers.__version__}` and `tokenizers=={tokenizers.__version__}`; "
                f"there might be inference-time regressions due to dependency changes. If in doubt, please"
                f"use the above versions."
            )

        # Instantiate PrismaticVisionBackbone (w/ Potential Fused Backbone)
        self.vision_backbone = PrismaticVisionBackbone(
            config.use_fused_vision_backbone, config.image_sizes, config.timm_model_ids, config.timm_override_act_layers
        )

        # Create Multimodal Projector
        self.projector = PrismaticProjector(
            config.use_fused_vision_backbone,
            vision_dim=self.vision_backbone.embed_dim,
            llm_dim=config.text_config.hidden_size,
        )

        # Instantiate LLM Backbone
        self.language_model = AutoModelForCausalLM.from_config(
            config.text_config, attn_implementation=config._attn_implementation
        )
        self.vocab_size = config.text_config.vocab_size
        self.pad_token_id = config.pad_token_id

        # HF Boilerplate =>> initializes weights via `_init_weights()` and sets gradient checkpointing
        self.post_init()

    # === `PreTrainedModel` Boilerplate ===
    def get_input_embeddings(self) -> nn.Module:
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Module:
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        self.language_model.set_output_embeddings(new_embeddings)

    def get_decoder(self) -> nn.Module:
        return self.language_model.get_decoder()

    def set_decoder(self, decoder: nn.Module) -> None:
        self.language_model.set_decoder(decoder)

    def tie_weights(self) -> None:
        self.language_model.tie_weights()  # Note: `Llama-2` and `Mistral` don't tie weights (no-op)

    def resize_token_embeddings(
        self, new_num_tokens: Optional[int] = None, pad_to_multiple_of: Optional[int] = None
    ) -> nn.Embedding:
        updated_embeddings = self.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)

        # Update config/instance variables
        self.config.text_config.vocab_size = updated_embeddings.num_embeddings
        self.vocab_size = updated_embeddings.num_embeddings

        return updated_embeddings

    # === Core Prismatic VLM `forward()` Logic ===
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_projector_features: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, PrismaticCausalLMOutputWithPast]:
        """Run a forward pass through the VLM, returning a PrismaticCausalLMOutputWithPast instance."""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_projector_features = output_projector_features if output_projector_features is not None else False
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Respect `use_cache` only if not training (even if `gradient_checkpointing` is off)
        use_cache = use_cache and not self.training

        # Instantiate Placeholder for Projector Features
        projected_patch_embeddings = None

        # Note :: We only support forward passes with the following cases:
        #   => Cached Generation :: (input_ids.shape[1] == 1) and (past_key_values is not None)
        #   => Unimodal Forward :: (pixel_values is None)
        #   => Multimodal Forward :: (pixel_values is not None) and (input_ids/embeds.shape[0] == pixel_values.shape[0])

        # === Handle Generation with Cache (`input_ids.shape[1] == 1`) =>> requires `past_keys_values` ===
        if input_ids.shape[1] == 1:
            assert input_ids.shape[0] == 1, "Generation is only currently supported for batch size of 1!"
            assert past_key_values is not None, "You must provide `past_key_values` during cached generation!"
            assert labels is None, "Unexpected key `labels` provided during cached generation!"

            language_model_output = self.language_model(
                input_ids=input_ids,
                attention_mask=None,
                position_ids=None,
                past_key_values=past_key_values,
                inputs_embeds=None,
                labels=None,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # === Handle Unimodal Forward ===
        elif pixel_values is None:
            assert (input_ids is not None) and (inputs_embeds is None), "Missing `input_ids` in language-only forward!"
            assert past_key_values is None, "Unexpected key `past_key_values` provided during language-only forward!"

            language_model_output = self.language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=None,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # === Handle Multimodal Forward ===
        elif (input_ids.shape[0] == pixel_values.shape[0]) or (inputs_embeds.shape[0] == pixel_values.shape[0]):
            assert past_key_values is None, "Unexpected key `past_key_values` provided during language-only forward!"

            # Visual Feature Extraction
            patch_features = self.vision_backbone(pixel_values)

            # Projection Logic =>> Update Attention Mask
            projected_patch_embeddings = self.projector(patch_features)
            projected_patch_attention_mask = None
            if attention_mask is not None:
                projected_patch_attention_mask = torch.full(
                    (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
                    fill_value=True,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )

            # Get Input Embeddings (from Language Model Embeddings)
            input_embeddings = self.get_input_embeddings()(input_ids)

            # Build Multimodal Embeddings & Attention Mask =>> Prismatic defaults to inserting after <BOS> token (1:)
            multimodal_embeddings = torch.cat(
                [input_embeddings[:, :1, :], projected_patch_embeddings, input_embeddings[:, 1:, :]], dim=1
            )
            multimodal_attention_mask = None
            if attention_mask is not None:
                multimodal_attention_mask = torch.cat(
                    [attention_mask[:, :1], projected_patch_attention_mask, attention_mask[:, 1:]], dim=1
                )

            # Build Labels (if specified) =>> Ignore Labels for Patch Embeddings
            multimodal_labels = None
            if labels is not None:
                projected_patch_labels = torch.full(
                    (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
                    fill_value=IGNORE_INDEX,
                    dtype=labels.dtype,
                    device=labels.device,
                )
                multimodal_labels = torch.cat([labels[:, :1], projected_patch_labels, labels[:, 1:]], dim=1)

            # Dispatch to Language Model
            language_model_output = self.language_model(
                input_ids=None,
                attention_mask=multimodal_attention_mask,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=multimodal_embeddings,
                labels=multimodal_labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # === Otherwise =>> Assume Invalid! ===
        elif (input_ids.shape[0] != pixel_values.shape[0]) or (inputs_embeds.shape[0] != pixel_values.shape[0]):
            raise ValueError("Non-homogenous batch of (text, image) input -- forward() does not support mixed batches!")

        else:
            raise ValueError(
                "Invalid PrismaticForConditionalGeneration `forward()` call with provided arguments:\n"
                f"=> `input_ids` = {input_ids is not None}\n"
                f"=> `attention_mask` = {attention_mask is not None}\n"
                f"=> `pixel_values` = {pixel_values is not None}\n"
                f"=> `labels` = {labels is not None}\n"
                f"=> `input_embeds` = {inputs_embeds is not None}\n"
                f"=> `past_key_values` = {past_key_values is not None}\n"
                f"=> `use_cache` = {use_cache}"
            )

        # Unpack `language_model_output` and return PrismaticCausalLMOutputWithPast (or tuple if not `return_dict`)
        if not return_dict:
            if output_projector_features and (projected_patch_embeddings is not None):
                return *language_model_output, projected_patch_embeddings

            return language_model_output

        return PrismaticCausalLMOutputWithPast(
            loss=language_model_output.loss,
            logits=language_model_output.logits,
            past_key_values=language_model_output.past_key_values,
            hidden_states=language_model_output.hidden_states,
            attentions=language_model_output.attentions,
            projector_features=projected_patch_embeddings,
        )

    # === GenerationMixin Methods ===
    def prepare_inputs_for_generation(
        self,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: str,
    ) -> Dict[str, torch.Tensor]:
        """Borrowed from `LlamaForCausalLM` and simplified for batch size = 1; mirrors original PrismaticVLM logic."""
        if ((input_ids is not None) and (input_ids.shape[0] > 1)) or (
            (inputs_embeds is not None) and (inputs_embeds.shape[0] > 1)
        ):
            raise ValueError("Generation with batch size > 1 is not currently supported!")

        # Handle `past_key_values` (cache) =>> assume `input_ids` just has unprocessed tokens
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        # If `input_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"input_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        # Make sure `pixel_values` are preserved in `model_inputs`
        model_inputs.update(
            {
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
            }
        )

        return model_inputs

    # Defer to Language Model (all handle this differently, with different return types)
    def _reorder_cache(self, *args, **kwargs) -> Any:
        return self.language_model._reorder_cache(*args, **kwargs)


class OpenVLAForActionPrediction(PrismaticForConditionalGeneration):
    """OpenVLA model for action prediction."""
    
    config_class = OpenVLAConfig
    
    def __init__(self, config):
        super().__init__(config)
        # Initialize model components here
        # This is a simplified implementation
        self.model = nn.Module()
        self.model.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
    
    def forward(self, input_ids=None, **kwargs):
        # Simplified forward pass
        return {"logits": torch.randn((input_ids.shape[0], input_ids.shape[1], self.config.vocab_size))}


class MixOfExperts(nn.Module):
    """Mixture of Experts layer."""
    
    def __init__(self, config, input_size, output_size):
        super().__init__()
        self.num_experts = config.num_experts
        self.num_selected_experts = config.num_selected_experts
        self.expert_dropout = config.expert_dropout
        self.load_balancing_loss_weight = config.load_balancing_loss_weight
        self.input_size = input_size
        self.output_size = output_size
        
        # Create expert networks
        self.experts = nn.ModuleList([
            nn.Linear(input_size, output_size) for _ in range(self.num_experts)
        ])
        
        # Router network to select experts
        self.router = nn.Linear(input_size, self.num_experts)
        
        # Initialize router weights
        nn.init.normal_(self.router.weight, mean=0.0, std=0.01)
        if hasattr(self.router, 'bias') and self.router.bias is not None:
            nn.init.zeros_(self.router.bias)
        
    def forward(self, x):
        """
        Forward pass through the MoE layer.
        
        Args:
            x: Input tensor of shape [batch_size, hidden_size]
            
        Returns:
            Tuple of (output, aux_loss)
        """
        batch_size = x.shape[0]
        
        # Get router logits
        router_logits = self.router(x)  # [batch_size, num_experts]
        
        # Add a small amount of noise to encourage exploration during training
        if self.training:
            router_logits += torch.randn_like(router_logits) * 1e-2
        
        # Get routing probabilities with softmax
        routing_weights = F.softmax(router_logits, dim=-1)  # [batch_size, num_experts]
        
        # Calculate auxiliary load balancing loss
        # We want each expert to be used equally across the batch
        aux_loss = self._calculate_load_balancing_loss(routing_weights)
        
        # Select top-k experts per token
        top_k_weights, top_k_indices = torch.topk(
            routing_weights, k=self.num_selected_experts, dim=-1
        )  # Both: [batch_size, num_selected_experts]
        
        # Normalize the weights
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        # Initialize output tensor
        final_output = torch.zeros(
            (batch_size, self.output_size), 
            device=x.device, 
            dtype=x.dtype
        )
        
        # For each expert, calculate its output and combine with weights
        for expert_idx in range(self.num_experts):
            # Find which batch items use this expert
            expert_mask = (top_k_indices == expert_idx).any(dim=-1)
            
            if not expert_mask.any():
                continue  # Skip if no batch items use this expert
                
            # Get the corresponding inputs
            expert_inputs = x[expert_mask]
            
            # Apply the expert
            expert_output = self.experts[expert_idx](expert_inputs)
            
            # For each batch item using this expert, find its weight
            for i, batch_idx in enumerate(torch.where(expert_mask)[0]):
                # Find which position this expert has in the top_k for this item
                expert_positions = (top_k_indices[batch_idx] == expert_idx).nonzero(as_tuple=True)[0]
                for pos in expert_positions:
                    weight = top_k_weights[batch_idx, pos]
                    final_output[batch_idx] += expert_output[i] * weight
        
        return final_output, aux_loss * self.load_balancing_loss_weight
        
    def _calculate_load_balancing_loss(self, routing_weights):
        """
        Calculate auxiliary load balancing loss to ensure all experts are used equally.
        
        Args:
            routing_weights: Routing weight tensor [batch_size, num_experts]
            
        Returns:
            Load balancing loss
        """
        # Fraction of tokens routed to each expert
        routing_prob = routing_weights.mean(dim=0)
        # Ideal routing probability (uniform)
        target_prob = torch.ones_like(routing_prob) / self.num_experts
        # L2 loss compared to uniform distribution
        return F.mse_loss(routing_prob, target_prob)
    
    def from_pretrained_layers(self, original_layer):
        """Initialize all experts with the pretrained weights."""
        for expert in self.experts:
            if hasattr(original_layer, 'weight'):
                expert.weight.data.copy_(original_layer.weight.data)
            if hasattr(original_layer, 'bias') and original_layer.bias is not None:
                expert.bias.data.copy_(original_layer.bias.data)


class MoEOpenVLAForActionPrediction(OpenVLAForActionPrediction):
    """MoE OpenVLA model for action prediction."""
    
    config_class = MoEOpenVLAConfig
    
    def __init__(self, config):
        super().__init__(config)
        
        # Add MoE layer - make sure it's compatible with your converted model
        self.action_moe = MixOfExperts(
            config,
            config.hidden_size,
            config.vocab_size
        )
        
        # Attribute to track MoE losses
        self.moe_loss = 0.0
        
    def forward(
        self, 
        input_ids=None,
        attention_mask=None,
        pixel_values=None,
        labels=None,
        **kwargs
    ):
        # Reset MoE loss
        self.moe_loss = 0.0
        
        # First get the regular model outputs
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels,
            **kwargs
        )
        
        # If there's no model.lm_head used in the original model, just return outputs
        if not hasattr(self, 'action_moe'):
            return outputs
            
        # Check if we can get hidden states - extract last token representation
        if hasattr(self.model, 'lm_head') and hasattr(outputs, 'logits'):
            # Extract the features before the lm_head
            batch_size = outputs.logits.shape[0]
            hidden_states = torch.zeros(
                (batch_size, self.config.hidden_size),
                device=outputs.logits.device,
                dtype=outputs.logits.dtype
            )
            
            # Get hidden states from model if available
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                # Last layer, last token
                if isinstance(outputs.hidden_states, tuple):
                    hidden_states = outputs.hidden_states[-1][:, -1, :]
                else:
                    hidden_states = outputs.hidden_states[:, -1, :]
            
            # Apply MoE layer
            moe_logits, aux_loss = self.action_moe(hidden_states)
            
            # Store the auxiliary loss
            self.moe_loss = aux_loss
            
            # Replace the logits for the last position with MoE logits
            if len(outputs.logits.shape) == 3:  # [batch, seq, vocab]
                # Only replace the last token logits
                outputs.logits[:, -1, :] = moe_logits
            else:  # [batch, vocab]
                outputs.logits = moe_logits
                
            # Update the loss if needed
            if hasattr(outputs, 'loss') and outputs.loss is not None:
                outputs.loss = outputs.loss + self.moe_loss
                
        return outputs
