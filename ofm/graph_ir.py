import torch
import torch.nn as nn
from typing import Dict, List, Any, Tuple, Optional
from collections import deque


class GraphIR:
    def __init__(self):
        self.graph = {}
        self.current_id = 0
        # To track the last node at each level for `next_module`
        self.previous_node_at_level = {}

    def _get_next_id(self) -> str:
        self.current_id += 1
        return f"node_{self.current_id}"

    def _get_module_dimensions(self, module: nn.Module) -> Tuple[Optional[List[int]], Optional[List[int]]]:
        input_dim = None
        output_dim = None

        if isinstance(module, nn.Linear):
            input_dim = [module.in_features]
            output_dim = [module.out_features]
        elif isinstance(module, nn.Conv2d):
            # Height and width are dynamic
            input_dim = [module.in_channels, None, None]
            output_dim = [module.out_channels, None, None]
        elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
            input_dim = list(module.normalized_shape)
            output_dim = list(module.normalized_shape)
        elif isinstance(module, nn.MultiheadAttention):
            input_dim = [module.embed_dim]
            output_dim = [module.embed_dim]
        # Add more conditions for other layer types as needed

        return input_dim, output_dim

    def _add_node(self, name: str, module: nn.Module, parent_id: str = None, level: int = 0) -> str:
        node_id = self._get_next_id()
        input_dim, output_dim = self._get_module_dimensions(module)

        self.graph[node_id] = {
            "name": name,
            "type": type(module).__name__,
            "children": [],
            "parent": parent_id,
            "next_module": None,  # This will be updated later
            "params": {param_name: param.size() for param_name, param in module.named_parameters(recurse=False)},
            "input_dim": input_dim,
            "output_dim": output_dim
        }

        # If there is a parent, add this node as a child of the parent
        if parent_id:
            self.graph[parent_id]["children"].append(node_id)

        # Handle `next_module` logic
        if level in self.previous_node_at_level:
            previous_node = self.previous_node_at_level[level]
            # Set the `next_module` of the previous sibling
            self.graph[previous_node]["next_module"] = node_id

        # Update the last node at this level to the current one
        self.previous_node_at_level[level] = node_id

        return node_id

    def _process_module(self, name: str, module: nn.Module, parent_id: str = None, level: int = 0) -> str:
        node_id = self._add_node(name, module, parent_id, level)

        if isinstance(module, (nn.Sequential, nn.ModuleList)):
            for idx, child in enumerate(module):
                child_name = f"{idx}" if isinstance(
                    module, nn.ModuleList) else f"{name}_{idx}"
                self._process_module(child_name, child, node_id, level + 1)
        else:
            for child_name, child_module in module.named_children():
                self._process_module(
                    child_name, child_module, node_id, level + 1)

        # Update `output_dim` to be the last child's `output_dim` if there are children
        if self.graph[node_id]["children"]:
            last_child_id = self.graph[node_id]["children"][-1]
            self.graph[node_id]["output_dim"] = self.graph[last_child_id]["output_dim"]

        # Update `input_dim` to be the parent's `input_dim` if available
        if parent_id:
            parent_output_dim = self.graph[parent_id]["output_dim"]
            self.graph[node_id]["input_dim"] = parent_output_dim

        return node_id

    def verify_graph_connectivity(self, graph: Dict[str, Any], start_node: str) -> bool:
        """
        Verifies that all nodes in the graph are reachable from the start node.
        """
        visited = set()
        queue = deque([start_node])

        while queue:
            node_id = queue.popleft()
            if node_id not in visited:
                visited.add(node_id)
                queue.extend(graph[node_id]['children'])

        return len(visited) == len(graph)

    def verify_dimension_compatibility(self, graph: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Verifies dimension compatibility between connected nodes.
        Returns a tuple: (is_compatible, list_of_incompatible_connections)
        """
        incompatible_connections = []

        for node_id, node in graph.items():
            output_dim = node['output_dim']
            for child_id in node['children']:
                child_node = graph[child_id]
                input_dim = child_node['input_dim']

                if not self._are_dimensions_compatible(output_dim, input_dim):
                    incompatible_connections.append(
                        f"{node_id} ({node['name']}) -> {child_id} ({child_node['name']})")

        return len(incompatible_connections) == 0, incompatible_connections

    def _are_dimensions_compatible(self, output_dim: List[int], input_dim: List[int]) -> bool:
        """
        Checks if output dimensions of one layer are compatible with input dimensions of the next layer.
        """
        if output_dim is None or input_dim is None:
            return True  # We can't verify, so we assume it's compatible

        if len(output_dim) != len(input_dim):
            return False

        for out_d, in_d in zip(output_dim, input_dim):
            if out_d is not None and in_d is not None and out_d != in_d:
                return False

        return True

    def verify_subnet(self, graph: Dict[str, Any], subnet_nodes: List[str]) -> Tuple[bool, str]:
        """
        Verifies if a given subnet is valid.
        Returns a tuple: (is_valid, error_message)
        """
        # Create a subgraph with only the selected nodes
        subgraph = {node_id: graph[node_id].copy() for node_id in subnet_nodes}
        for node in subgraph.values():
            node['children'] = [
                child for child in node['children'] if child in subnet_nodes]

        # Verify connectivity
        # Assuming the first node is the start node
        start_node = subnet_nodes[0]
        if not self.verify_graph_connectivity(subgraph, start_node):
            return False, "Subnet is not fully connected"

        # Verify dimension compatibility
        is_compatible, incompatible_connections = self.verify_dimension_compatibility(
            subgraph)
        if not is_compatible:
            return False, f"Dimension mismatch in connections: {', '.join(incompatible_connections)}"

        return True, "Subnet is valid"

    def convert(self, model: nn.Module) -> Dict[str, Any]:
        self.graph = {}
        self.current_id = 0
        self.previous_node_at_level = {}  # Reset level tracking for each conversion
        self._process_module("root", model)
        return self.graph

    def print_graph(self, node_id: str = "node_1", depth: int = 0):
        node = self.graph[node_id]
        print("  " * depth + f"{node['name']} ({node['type']})")
        for child_id in node["children"]:
            self.print_graph(child_id, depth + 1)

# def convert_model_to_graph(model: nn.Module) -> Dict[str, Any]:
#     converter = ModelToGraphConverter()
#     return converter.convert(model)
