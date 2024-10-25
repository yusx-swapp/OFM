import torch.nn as nn
from collections import OrderedDict
from typing import Dict, Any, Optional
from copy import deepcopy

class GraphIR:
    def __init__(self, model: nn.Module):
        self.model = model
        # Store original weights
        self.weights_dict = OrderedDict(model.state_dict())
        # Main metadata dictionary
        self.metadata_dict = OrderedDict()
        # Dictionary to store new configurations for elastic modules
        self.elastic_config_dict = OrderedDict()
        # Build the IR
        self._build_ir()

    def _build_ir(self):
        """Build metadata dictionary for user-defined modules."""
        for name, module in self.model.named_modules():
            # Skip built-in PyTorch modules
            if type(module).__module__.startswith('torch.nn'):
                continue
            
            metadata = self._create_module_metadata(name, module)
            if metadata:  # Only add if we got valid metadata
                self.metadata_dict[name] = metadata
                # Initialize empty elastic config
                self.elastic_config_dict[name] = {}

    def _create_module_metadata(self, name: str, module: nn.Module) -> Optional[Dict[str, Any]]:
        """Create metadata for a user-defined module."""
        try:
            # Get module's __init__ signature
            import inspect
            init_signature = inspect.signature(module.__class__.__init__)
            
            # Get current parameter values where possible
            current_args = {}
            for param_name, param in init_signature.parameters.items():
                if param_name == 'self':
                    continue
                # Try to get the current value from module attributes
                try:
                    current_args[param_name] = getattr(module, param_name)
                except AttributeError:
                    current_args[param_name] = None

            metadata = {
                'module_info': {
                    'type': type(module).__name__,
                    'path': f"{module.__class__.__module__}.{module.__class__.__name__}",
                },
                'init_args': current_args,
                'elastic': False  # Default to non-elastic
            }
            return metadata
        except Exception as e:
            print(f"Warning: Could not create metadata for module {name}: {e}")
            return None


    def set_elastic_config(self, module_name: str, config: Dict[str, ElasticRange]):
        """Set elastic configuration for a module."""
        if module_name not in self.metadata_dict:
            raise KeyError(f"Module {module_name} not found")
        
        self.metadata_dict[module_name]['elastic'] = True
        self.elastic_config_dict[module_name] = config

    def sample_elastic_config(self, module_name: str) -> Dict[str, Any]:
        """Sample new configuration for an elastic module."""
        if not self.elastic_config_dict[module_name]:
            raise ValueError(f"No elastic config set for {module_name}")
        
        sampled_config = {}
        for param_name, range_obj in self.elastic_config_dict[module_name].items():
            sampled_config[param_name] = range_obj.sample()
        return sampled_config


    def update_elastic_config(self, module_name: str, new_config: Dict[str, Any]):
        """Update the new configuration for an elastic module."""
        if module_name not in self.metadata_dict:
            raise KeyError(f"Module {module_name} not found in metadata dictionary")
        
        if not self.metadata_dict[module_name]['elastic']:
            raise ValueError(f"Module {module_name} is not marked as elastic")
        
        self.elastic_config_dict[module_name] = new_config

    def get_module_metadata(self, name: str) -> Dict[str, Any]:
        """Get metadata for a specific module."""
        return self.metadata_dict.get(name)

    def print_metadata_dict(self, indent=2):
        """Pretty print the metadata dictionary."""
        def _format_dict(d, level=0):
            output = ""
            for key, value in d.items():
                space = " " * (level * indent)
                if isinstance(value, dict):
                    output += f"{space}{key}:\n{_format_dict(value, level + 1)}"
                else:
                    output += f"{space}{key}: {value}\n"
            return output

        print("\nMetadata Dictionary:")
        for module_name, metadata in self.metadata_dict.items():
            print(f"\n{'='*50}")
            print(f"Module: {module_name}")
            print(_format_dict(metadata))
            if self.elastic_config_dict[module_name]:
                print("\nElastic Config:")
                print(_format_dict(self.elastic_config_dict[module_name]))
    