#!/usr/bin/env python3
# -----------------------------------------------------------------------------
#
# Layer-wise ONNX Export Experiment
# Exploring torch.onnx.dynamo_export for layer-by-layer model export
#
# -----------------------------------------------------------------------------

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

import torch
import torch.nn as nn
import onnx
from onnx import helper, numpy_helper
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LayerWrapper(nn.Module):
    """
    Wrapper to export a single transformer layer with proper I/O handling.
    """
    def __init__(
        self,
        layer: nn.Module,
        layer_idx: int,
        config,
        include_embeddings: bool = False,
        include_lm_head: bool = False,
        embed_tokens: Optional[nn.Module] = None,
        lm_head: Optional[nn.Module] = None,
        norm: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.layer = layer
        self.layer_idx = layer_idx
        self.config = config
        self.include_embeddings = include_embeddings
        self.include_lm_head = include_lm_head
        
        if include_embeddings and embed_tokens is not None:
            self.embed_tokens = embed_tokens
        if include_lm_head and lm_head is not None:
            self.lm_head = lm_head
        if include_lm_head and norm is not None:
            self.norm = norm
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,  # Disable KV cache for simpler export
    ):
        """
        Forward pass for a single layer.
        
        Args:
            hidden_states: Input hidden states [batch, seq_len, hidden_dim]
            attention_mask: Attention mask [batch, 1, seq_len, seq_len]
            position_ids: Position IDs [batch, seq_len]
            past_key_value: Tuple of (key, value) tensors for KV cache
            use_cache: Whether to return updated cache
        
        Returns:
            hidden_states tensor
        """
        # Apply layer without KV cache for simpler export
        layer_outputs = self.layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=None,  # Disable KV cache
            use_cache=False,
        )
        
        # Extract outputs - always first element
        if isinstance(layer_outputs, tuple):
            hidden_states = layer_outputs[0]
        else:
            hidden_states = layer_outputs
        
        # Apply final norm and LM head if this is the last layer
        if self.include_lm_head:
            if hasattr(self, 'norm'):
                hidden_states = self.norm(hidden_states)
            if hasattr(self, 'lm_head'):
                logits = self.lm_head(hidden_states)
                return logits
        
        return hidden_states


class LayerWiseONNXExporter:
    """
    Main class for layer-wise ONNX export experiment.
    """
    
    def __init__(
        self,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        export_dir: str = "./layer_wise_export",
        use_dynamo: bool = True,
    ):
        self.model_name = model_name
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self.use_dynamo = use_dynamo
        
        logger.info(f"Loading model: {model_name}")
        self.config = AutoConfig.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="cpu",
        )
        self.model.eval()
        
        # Extract model components
        self.num_layers = self.config.num_hidden_layers
        logger.info(f"Model has {self.num_layers} transformer layers")
        
        # Get model architecture specific attributes
        self._extract_model_components()
    
    def _extract_model_components(self):
        """Extract model-specific components (layers, embeddings, etc.)"""
        # This is model-architecture specific
        # Handle different model types (LLaMA, GPT, etc.)
        
        if hasattr(self.model, 'model'):
            # LLaMA-style models
            self.base_model = self.model.model
            self.layers = self.base_model.layers
            self.embed_tokens = self.base_model.embed_tokens
            self.norm = self.base_model.norm
            self.lm_head = self.model.lm_head
        elif hasattr(self.model, 'transformer'):
            # GPT-style models
            self.base_model = self.model.transformer
            self.layers = self.base_model.h
            self.embed_tokens = self.base_model.wte
            self.norm = self.base_model.ln_f
            self.lm_head = self.model.lm_head
        else:
            raise ValueError(f"Unsupported model architecture: {type(self.model)}")
        
        logger.info(f"Extracted {len(self.layers)} layers from model")
    
    def create_example_inputs(
        self,
        batch_size: int = 1,
        seq_len: int = 8,
        past_seq_len: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """
        Create example inputs for ONNX export.
        
        Args:
            batch_size: Batch size
            seq_len: Sequence length
            past_seq_len: Past sequence length for KV cache
        
        Returns:
            Dictionary of input tensors
        """
        hidden_size = self.config.hidden_size
        num_heads = self.config.num_attention_heads
        head_dim = hidden_size // num_heads
        
        inputs = {
            'hidden_states': torch.randn(batch_size, seq_len, hidden_size),
            'attention_mask': torch.ones(batch_size, 1, seq_len, seq_len + past_seq_len),
            'position_ids': torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1),
        }
        
        # Add past_key_value if needed
        if past_seq_len > 0:
            # KV cache format: (key, value) each of shape [batch, num_heads, past_seq_len, head_dim]
            past_key = torch.randn(batch_size, num_heads, past_seq_len, head_dim)
            past_value = torch.randn(batch_size, num_heads, past_seq_len, head_dim)
            inputs['past_key_value'] = (past_key, past_value)
        
        return inputs
    
    def export_layer_traditional(
        self,
        layer_idx: int,
        include_embeddings: bool = False,
        include_lm_head: bool = False,
    ) -> Path:
        """
        Export a single layer using traditional torch.onnx.export.
        
        Args:
            layer_idx: Index of the layer to export
            include_embeddings: Whether to include embedding layer
            include_lm_head: Whether to include LM head
        
        Returns:
            Path to exported ONNX file
        """
        logger.info(f"Exporting layer {layer_idx} using traditional torch.onnx.export")
        
        # Create layer wrapper
        layer_wrapper = LayerWrapper(
            layer=self.layers[layer_idx],
            layer_idx=layer_idx,
            config=self.config,
            include_embeddings=include_embeddings,
            include_lm_head=include_lm_head,
            embed_tokens=self.embed_tokens if include_embeddings else None,
            lm_head=self.lm_head if include_lm_head else None,
            norm=self.norm if include_lm_head else None,
        )
        layer_wrapper.eval()
        
        # Create example inputs
        example_inputs = self.create_example_inputs(batch_size=1, seq_len=8, past_seq_len=0)
        
        # Prepare inputs for export
        hidden_states = example_inputs['hidden_states']
        attention_mask = example_inputs['attention_mask']
        position_ids = example_inputs['position_ids']
        
        # Define input/output names
        input_names = ['hidden_states', 'attention_mask', 'position_ids']
        
        if include_lm_head:
            output_names = ['logits']
        else:
            output_names = ['hidden_states_out']
        
        # Define dynamic axes
        dynamic_axes = {
            'hidden_states': {0: 'batch_size', 1: 'seq_len'},
            'attention_mask': {0: 'batch_size', 2: 'seq_len', 3: 'total_seq_len'},
            'position_ids': {0: 'batch_size', 1: 'seq_len'},
        }
        
        if not include_lm_head:
            dynamic_axes['hidden_states_out'] = {0: 'batch_size', 1: 'seq_len'}
        else:
            dynamic_axes['logits'] = {0: 'batch_size', 1: 'seq_len'}
        
        # Export path
        onnx_path = self.export_dir / f"layer_{layer_idx}_traditional.onnx"
        
        # Export
        with torch.no_grad():
            torch.onnx.export(
                layer_wrapper,
                (hidden_states, attention_mask, position_ids, None, False),
                str(onnx_path),
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=17,
                do_constant_folding=True,
            )
        
        logger.info(f"Layer {layer_idx} exported to {onnx_path}")
        return onnx_path
    
    def export_layer_dynamo(
        self,
        layer_idx: int,
        include_embeddings: bool = False,
        include_lm_head: bool = False,
        ir_version: int = 10,
    ) -> Path:
        """
        Export a single layer using torch.onnx.dynamo_export.
        
        Args:
            layer_idx: Index of the layer to export
            include_embeddings: Whether to include embedding layer
            include_lm_head: Whether to include LM head
            ir_version: ONNX IR version (default 10 for compatibility)
        
        Returns:
            Path to exported ONNX file
        """
        logger.info(f"Exporting layer {layer_idx} using torch.onnx.dynamo_export")
        
        # Check if dynamo_export is available
        if not hasattr(torch.onnx, 'dynamo_export'):
            logger.warning("torch.onnx.dynamo_export not available. Falling back to traditional export.")
            return self.export_layer_traditional(layer_idx, include_embeddings, include_lm_head)
        
        # Create layer wrapper
        layer_wrapper = LayerWrapper(
            layer=self.layers[layer_idx],
            layer_idx=layer_idx,
            config=self.config,
            include_embeddings=include_embeddings,
            include_lm_head=include_lm_head,
            embed_tokens=self.embed_tokens if include_embeddings else None,
            lm_head=self.lm_head if include_lm_head else None,
            norm=self.norm if include_lm_head else None,
        )
        layer_wrapper.eval()
        
        # Create example inputs
        example_inputs = self.create_example_inputs(batch_size=1, seq_len=8, past_seq_len=0)
        
        # Prepare inputs for export
        hidden_states = example_inputs['hidden_states']
        attention_mask = example_inputs['attention_mask']
        position_ids = example_inputs['position_ids']
        
        # Export path
        onnx_path = self.export_dir / f"layer_{layer_idx}_dynamo.onnx"
        
        # Export using dynamo
        try:
            with torch.no_grad():
                onnx_program = torch.onnx.dynamo_export(
                    layer_wrapper,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,  # past_key_value
                    True,  # use_cache
                )
                # Save with specified IR version for compatibility
                onnx_program.save(str(onnx_path))
                
                # Load and modify IR version if needed
                if ir_version != 11:  # Default dynamo IR version is 11
                    model = onnx.load(str(onnx_path))
                    model.ir_version = ir_version
                    onnx.save(model, str(onnx_path))
                    logger.info(f"Modified IR version to {ir_version} for compatibility")
            
            logger.info(f"Layer {layer_idx} exported to {onnx_path}")
            return onnx_path
        
        except Exception as e:
            logger.error(f"Dynamo export failed: {e}")
            logger.info("Falling back to traditional export")
            return self.export_layer_traditional(layer_idx, include_embeddings, include_lm_head)
    
    def export_all_layers(self, ir_version: int = 10) -> List[Path]:
        """
        Export all layers individually.
        
        Args:
            ir_version: ONNX IR version (default 10 for compatibility)
        
        Returns:
            List of paths to exported ONNX files
        """
        logger.info(f"Starting layer-wise export for {self.num_layers} layers")
        
        exported_paths = []
        
        for layer_idx in range(self.num_layers):
            # First layer includes embeddings, last layer includes LM head
            include_embeddings = (layer_idx == 0)
            include_lm_head = (layer_idx == self.num_layers - 1)
            
            if self.use_dynamo:
                path = self.export_layer_dynamo(
                    layer_idx,
                    include_embeddings=include_embeddings,
                    include_lm_head=include_lm_head,
                    ir_version=ir_version,
                )
            else:
                path = self.export_layer_traditional(
                    layer_idx,
                    include_embeddings=include_embeddings,
                    include_lm_head=include_lm_head,
                )
            
            exported_paths.append(path)
        
        logger.info(f"All {self.num_layers} layers exported successfully")
        return exported_paths
    
    def combine_onnx_models(self, layer_paths: List[Path]) -> Path:
        """
        Combine individual layer ONNX models into a single model.
        
        This is a complex operation that involves:
        1. Loading all layer models
        2. Renaming nodes to avoid conflicts
        3. Connecting outputs of layer N to inputs of layer N+1
        4. Merging all graphs into one
        
        Args:
            layer_paths: List of paths to layer ONNX files
        
        Returns:
            Path to combined ONNX model
        """
        logger.info("Starting ONNX model combination")
        
        # Load all models
        models = []
        for path in layer_paths:
            model = onnx.load(str(path))
            models.append(model)
            logger.info(f"Loaded {path.name}")
        
        # This is a simplified approach - full implementation would need:
        # 1. Graph merging with proper node renaming
        # 2. Tensor name conflict resolution
        # 3. Proper input/output connection
        # 4. Initializer merging
        
        logger.warning("ONNX model combination is complex and requires careful graph manipulation")
        logger.warning("This is a placeholder - full implementation would use onnx.compose or manual graph merging")
        
        # For now, just save a reference to the approach
        combined_path = self.export_dir / "combined_model.onnx"
        
        # TODO: Implement actual combination logic
        # This would involve:
        # - Creating a new ModelProto
        # - Merging all graphs
        # - Connecting layer outputs to next layer inputs
        # - Handling KV cache routing
        
        logger.info(f"Combined model would be saved to {combined_path}")
        return combined_path
    
    def validate_layer_export(self, layer_idx: int, onnx_path: Path):
        """
        Validate that exported layer produces correct outputs.
        
        Args:
            layer_idx: Index of the layer
            onnx_path: Path to exported ONNX file
        """
        logger.info(f"Validating layer {layer_idx} export")
        
        try:
            import onnxruntime as ort
            
            # Create test inputs
            example_inputs = self.create_example_inputs(batch_size=1, seq_len=8)
            
            # Run PyTorch model
            layer_wrapper = LayerWrapper(
                layer=self.layers[layer_idx],
                layer_idx=layer_idx,
                config=self.config,
            )
            layer_wrapper.eval()
            
            with torch.no_grad():
                pt_outputs = layer_wrapper(
                    example_inputs['hidden_states'],
                    example_inputs['attention_mask'],
                    example_inputs['position_ids'],
                    None,
                    True,
                )
            
            # Run ONNX model
            session = ort.InferenceSession(str(onnx_path))
            ort_inputs = {
                'hidden_states': example_inputs['hidden_states'].numpy(),
                'attention_mask': example_inputs['attention_mask'].numpy(),
                'position_ids': example_inputs['position_ids'].numpy(),
            }
            ort_outputs = session.run(None, ort_inputs)
            
            # Compare outputs
            if isinstance(pt_outputs, tuple):
                pt_hidden = pt_outputs[0].numpy()
            else:
                pt_hidden = pt_outputs.numpy()
            
            ort_hidden = ort_outputs[0]
            
            max_diff = np.abs(pt_hidden - ort_hidden).max()
            logger.info(f"Layer {layer_idx} validation - Max difference: {max_diff}")
            
            if max_diff < 1e-5:
                logger.info(f"✓ Layer {layer_idx} validation PASSED")
            else:
                logger.warning(f"✗ Layer {layer_idx} validation FAILED - difference too large")
        
        except Exception as e:
            logger.error(f"Validation failed: {e}")


def main():
    """Main experiment runner"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Layer-wise ONNX Export Experiment")
    parser.add_argument(
        "--model-name",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--export-dir",
        type=str,
        default="./layer_wise_export",
        help="Directory to save exported models"
    )
    parser.add_argument(
        "--use-dynamo",
        action="store_true",
        help="Use torch.onnx.dynamo_export instead of traditional export"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate exported models"
    )
    parser.add_argument(
        "--combine",
        action="store_true",
        help="Attempt to combine exported layers"
    )
    
    args = parser.parse_args()
    
    # Create exporter
    exporter = LayerWiseONNXExporter(
        model_name=args.model_name,
        export_dir=args.export_dir,
        use_dynamo=args.use_dynamo,
    )
    
    # Export all layers
    layer_paths = exporter.export_all_layers()
    
    # Validate if requested
    if args.validate:
        for idx, path in enumerate(layer_paths):
            exporter.validate_layer_export(idx, path)
    
    # Combine if requested
    if args.combine:
        combined_path = exporter.combine_onnx_models(layer_paths)
        logger.info(f"Combined model path: {combined_path}")
    
    logger.info("Experiment completed!")
    logger.info(f"Exported {len(layer_paths)} layer models to {args.export_dir}")


if __name__ == "__main__":
    main()
