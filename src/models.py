"""Models registry for Bedrock Latency Benchmark.

Defines available lightweight models in ap-northeast-2 (Seoul) region.
Note: Some models require cross-region inference profiles instead of direct model IDs.
"""

from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

# Lightweight models available in ap-northeast-2 (Seoul) region
# Note: Nova models require APAC inference profiles for on-demand usage in Seoul region
# Claude 3.5 Haiku requires Global inference profile (no APAC profile available)
LIGHTWEIGHT_MODELS: Dict[str, Dict[str, str]] = {
    "claude-3-haiku": {
        "model_id": "anthropic.claude-3-haiku-20240307-v1:0",
        "provider": "Anthropic",
        "description": "Fast, cost-effective (direct access in Seoul)",
    },
    "claude-3.5-haiku": {
        # Claude 3.5 Haiku requires Global inference profile (no APAC profile)
        "model_id": "global.anthropic.claude-haiku-4-5-20251001-v1:0",
        "provider": "Anthropic",
        "description": "Claude Haiku 4.5 (via Global inference profile)",
    },
    "nova-micro": {
        # Nova Micro requires APAC inference profile in Seoul region
        "model_id": "apac.amazon.nova-micro-v1:0",
        "provider": "Amazon",
        "description": "Text-only, lowest latency (via APAC inference profile)",
    },
    "nova-lite": {
        # Nova Lite requires APAC inference profile in Seoul region
        "model_id": "apac.amazon.nova-lite-v1:0",
        "provider": "Amazon",
        "description": "Multimodal, low cost (via APAC inference profile)",
    },
}


def get_available_models() -> Dict[str, Dict[str, str]]:
    """Get all available lightweight models.
    
    Returns:
        Dictionary of model names to model information
    """
    return LIGHTWEIGHT_MODELS.copy()


def get_model_info(model_name: str) -> Optional[Dict[str, str]]:
    """Get information for a specific model.
    
    Args:
        model_name: Short name of the model (e.g., 'claude-3-haiku')
        
    Returns:
        Model information dictionary or None if not found
    """
    return LIGHTWEIGHT_MODELS.get(model_name)


def get_model_id(model_name: str) -> Optional[str]:
    """Get the full model ID for a model name.
    
    Args:
        model_name: Short name of the model
        
    Returns:
        Full model ID string or None if not found
    """
    model_info = get_model_info(model_name)
    return model_info["model_id"] if model_info else None


def get_all_model_ids() -> List[str]:
    """Get list of all model IDs.
    
    Returns:
        List of full model ID strings
    """
    return [info["model_id"] for info in LIGHTWEIGHT_MODELS.values()]


def get_model_names() -> List[str]:
    """Get list of all model short names.
    
    Returns:
        List of model short names
    """
    return list(LIGHTWEIGHT_MODELS.keys())


def is_model_available(model_name: str) -> bool:
    """Check if a model is available in the registry.
    
    Args:
        model_name: Short name of the model
        
    Returns:
        True if model is available, False otherwise
    """
    return model_name in LIGHTWEIGHT_MODELS


def display_models() -> str:
    """Generate a formatted string displaying all available models.
    
    Returns:
        Formatted string with model information
    """
    lines = ["Available Lightweight Models (ap-northeast-2):"]
    lines.append("-" * 60)
    
    for name, info in LIGHTWEIGHT_MODELS.items():
        lines.append(f"  {name}:")
        lines.append(f"    Model ID: {info['model_id']}")
        lines.append(f"    Provider: {info['provider']}")
        lines.append(f"    Description: {info['description']}")
        lines.append("")
    
    return "\n".join(lines)
