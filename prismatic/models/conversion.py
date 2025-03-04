from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction, MoEOpenVLAForActionPrediction
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig, MoEOpenVLAConfig
from prismatic.overwatch import initialize_overwatch

# Add these functions to handle Flash Attention configuration
def create_model_config(config_class, **kwargs):
    """Create a model config with Flash Attention disabled"""
    config = config_class(**kwargs)
    # Disable Flash Attention
    config._attn_implementation = "eager"
    return config

def load_model_without_flash_attn(model_class, model_path, **kwargs):
    """Load a model with Flash Attention disabled"""
    kwargs["attn_implementation"] = "eager"
    return model_class.from_pretrained(model_path, **kwargs) 