"""
ONNX Model Analysis - Extract metadata, FLOPs, memory usage
"""
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import numpy as np

import onnx
from onnx import numpy_helper
import onnxruntime as ort

from src.utils.logger import logger
from src.utils.helpers import get_model_size_mb, format_size, count_parameters


class ONNXAnalyzer:
    """
    Comprehensive ONNX model analysis
    """
    
    def __init__(self, model_path: Path):
        """
        Initialize analyzer with ONNX model
        
        Args:
            model_path: Path to ONNX model file
        """
        self.model_path = Path(model_path)
        self.model = onnx.load(str(model_path))
        self.graph = self.model.graph
        
        # Extract initializers (weights)
        self.initializers = {init.name: init for init in self.graph.initializer}
        
        logger.info(f"Loaded ONNX model: {self.model_path.name}")
    
    def analyze(self) -> Dict[str, Any]:
        """
        Perform comprehensive model analysis
        
        Returns:
            Dict containing all analysis results
        """
        logger.info("Starting comprehensive model analysis...")
        
        analysis = {
            'model_info': self._get_model_info(),
            'architecture': self._analyze_architecture(),
            'parameters': self._analyze_parameters(),
            'operations': self._analyze_operations(),
            'memory': self._analyze_memory(),
            'computation': self._analyze_computation(),
            'layer_details': self._get_layer_details(),
            'optimization_potential': self._assess_optimization_potential()
        }
        
        logger.info("✓ Model analysis complete")
        
        return analysis
    
    def _get_model_info(self) -> Dict[str, Any]:
        """Get basic model information"""
        info = {
            'model_name': self.model_path.stem,
            'model_path': str(self.model_path),
            'model_size_mb': get_model_size_mb(self.model_path),
            'opset_version': self.model.opset_import[0].version if self.model.opset_import else None,
            'producer_name': self.model.producer_name,
            'producer_version': self.model.producer_version,
        }
        
        # Input/Output information
        inputs = []
        for input_tensor in self.graph.input:
            if input_tensor.name not in self.initializers:
                shape = [dim.dim_value if dim.dim_value > 0 else -1 
                        for dim in input_tensor.type.tensor_type.shape.dim]
                inputs.append({
                    'name': input_tensor.name,
                    'shape': shape,
                    'dtype': onnx.TensorProto.DataType.Name(
                        input_tensor.type.tensor_type.elem_type
                    )
                })
        
        outputs = []
        for output_tensor in self.graph.output:
            shape = [dim.dim_value if dim.dim_value > 0 else -1 
                    for dim in output_tensor.type.tensor_type.shape.dim]
            outputs.append({
                'name': output_tensor.name,
                'shape': shape,
                'dtype': onnx.TensorProto.DataType.Name(
                    output_tensor.type.tensor_type.elem_type
                )
            })
        
        info['inputs'] = inputs
        info['outputs'] = outputs
        
        return info
    
    def _analyze_architecture(self) -> Dict[str, Any]:
        """Analyze model architecture"""
        nodes = self.graph.node
        
        # Count operator types
        op_counts = defaultdict(int)
        for node in nodes:
            op_counts[node.op_type] += 1
        
        # Determine architecture type
        arch_type = self._infer_architecture_type(op_counts)
        
        architecture = {
            'total_nodes': len(nodes),
            'operator_types': dict(op_counts),
            'architecture_type': arch_type,
            'depth': self._calculate_graph_depth(),
        }
        
        return architecture
    
    def _infer_architecture_type(self, op_counts: Dict[str, int]) -> str:
        """Infer model architecture type from operators"""
        # Simple heuristic-based inference
        if op_counts.get('Conv', 0) > 10:
            if op_counts.get('MatMul', 0) > 5 or op_counts.get('Attention', 0) > 0:
                return 'Hybrid CNN-Transformer'
            return 'CNN (Convolutional Neural Network)'
        
        elif op_counts.get('MatMul', 0) > 10 or op_counts.get('Gemm', 0) > 10:
            if op_counts.get('LSTM', 0) > 0 or op_counts.get('GRU', 0) > 0:
                return 'RNN (Recurrent Neural Network)'
            elif op_counts.get('Attention', 0) > 0:
                return 'Transformer'
            return 'MLP (Multi-Layer Perceptron)'
        
        elif op_counts.get('LSTM', 0) > 0 or op_counts.get('GRU', 0) > 0:
            return 'RNN (Recurrent Neural Network)'
        
        return 'Custom Architecture'
    
    def _calculate_graph_depth(self) -> int:
        """Calculate maximum depth of computation graph"""
        # Build adjacency list
        node_outputs = {}
        node_inputs = {}
        
        for node in self.graph.node:
            node_outputs[node.name] = node.output
            node_inputs[node.name] = node.input
        
        # Simple depth calculation (can be improved)
        return len(self.graph.node)  # Simplified version
    
    def _analyze_parameters(self) -> Dict[str, Any]:
        """Analyze model parameters"""
        params_dict = {}
        
        for name, init in self.initializers.items():
            tensor = numpy_helper.to_array(init)
            params_dict[name] = tensor
        
        total_params = count_parameters(params_dict)
        
        # Calculate parameter sizes by type
        param_types = defaultdict(int)
        param_shapes = {}
        
        for name, tensor in params_dict.items():
            param_types[str(tensor.dtype)] += tensor.size
            param_shapes[name] = tensor.shape
        
        parameters = {
            'total_parameters': total_params,
            'total_parameters_millions': total_params / 1e6,
            'parameter_types': dict(param_types),
            'trainable_parameters': total_params,  # ONNX doesn't distinguish
            'parameter_shapes': param_shapes,
        }
        
        return parameters
    
    def _analyze_operations(self) -> Dict[str, Any]:
        """Analyze operations in the model"""
        operations = {
            'total_operations': len(self.graph.node),
            'operation_breakdown': defaultdict(list)
        }
        
        for node in self.graph.node:
            operations['operation_breakdown'][node.op_type].append({
                'name': node.name,
                'inputs': list(node.input),
                'outputs': list(node.output)
            })
        
        # Convert defaultdict to dict
        operations['operation_breakdown'] = dict(operations['operation_breakdown'])
        
        return operations
    
    def _analyze_memory(self) -> Dict[str, Any]:
        """Analyze memory requirements"""
        # Calculate weight memory
        weight_memory = 0
        for init in self.initializers.values():
            tensor = numpy_helper.to_array(init)
            weight_memory += tensor.nbytes
        
        # Estimate activation memory (simplified)
        activation_memory = self._estimate_activation_memory()
        
        memory = {
            'weight_memory_bytes': weight_memory,
            'weight_memory_mb': weight_memory / (1024 * 1024),
            'estimated_activation_memory_bytes': activation_memory,
            'estimated_activation_memory_mb': activation_memory / (1024 * 1024),
            'peak_memory_mb': (weight_memory + activation_memory) / (1024 * 1024),
        }
        
        return memory
    
    def _estimate_activation_memory(self) -> int:
        """Estimate activation memory (simplified)"""
        # This is a simplified estimation
        # In practice, need to trace through graph with actual shapes
        
        total_activation = 0
        
        # Estimate from Conv and MatMul operations
        for node in self.graph.node:
            if node.op_type in ['Conv', 'MatMul', 'Gemm']:
                # Rough estimate: assume float32 (4 bytes per element)
                # This needs proper shape inference for accuracy
                total_activation += 1024 * 1024  # Placeholder: 1MB per op
        
        return total_activation
    
    def _analyze_computation(self) -> Dict[str, Any]:
        """Analyze computational requirements (FLOPs)"""
        total_flops = 0
        layer_flops = {}
        
        for node in self.graph.node:
            flops = self._estimate_node_flops(node)
            total_flops += flops
            if flops > 0:
                layer_flops[node.name] = {
                    'op_type': node.op_type,
                    'flops': flops,
                    'gflops': flops / 1e9
                }
        
        computation = {
            'total_flops': total_flops,
            'total_gflops': total_flops / 1e9,
            'layer_flops': layer_flops,
        }
        
        return computation
    
    def _estimate_node_flops(self, node: onnx.NodeProto) -> int:
        """Estimate FLOPs for a single node"""
        # Simplified FLOP estimation
        # For accurate results, need shape inference
        
        if node.op_type == 'Conv':
            # Conv FLOPs = 2 * C_in * C_out * K_h * K_w * H_out * W_out
            # Simplified: return rough estimate
            return 100_000_000  # Placeholder
        
        elif node.op_type in ['MatMul', 'Gemm']:
            # MatMul FLOPs = 2 * M * N * K
            return 10_000_000  # Placeholder
        
        elif node.op_type in ['Relu', 'Sigmoid', 'Tanh']:
            return 1_000_000  # Placeholder
        
        return 0
    
    def _get_layer_details(self) -> List[Dict[str, Any]]:
        """Get detailed information for each layer"""
        layers = []
        
        for i, node in enumerate(self.graph.node):
            layer = {
                'index': i,
                'name': node.name,
                'op_type': node.op_type,
                'inputs': list(node.input),
                'outputs': list(node.output),
                'attributes': {}
            }
            
            # Extract attributes
            for attr in node.attribute:
                layer['attributes'][attr.name] = self._get_attribute_value(attr)
            
            # Get weight info if available
            weight_info = {}
            for input_name in node.input:
                if input_name in self.initializers:
                    tensor = numpy_helper.to_array(self.initializers[input_name])
                    weight_info[input_name] = {
                        'shape': tensor.shape,
                        'dtype': str(tensor.dtype),
                        'size': tensor.size,
                        'memory_mb': tensor.nbytes / (1024 * 1024)
                    }
            
            if weight_info:
                layer['weights'] = weight_info
            
            layers.append(layer)
        
        return layers
    
    def _get_attribute_value(self, attr: onnx.AttributeProto) -> Any:
        """Extract attribute value from ONNX attribute"""
        if attr.type == onnx.AttributeProto.FLOAT:
            return attr.f
        elif attr.type == onnx.AttributeProto.INT:
            return attr.i
        elif attr.type == onnx.AttributeProto.STRING:
            return attr.s.decode('utf-8')
        elif attr.type == onnx.AttributeProto.TENSOR:
            return numpy_helper.to_array(attr.t)
        elif attr.type == onnx.AttributeProto.FLOATS:
            return list(attr.floats)
        elif attr.type == onnx.AttributeProto.INTS:
            return list(attr.ints)
        elif attr.type == onnx.AttributeProto.STRINGS:
            return [s.decode('utf-8') for s in attr.strings]
        else:
            return None
    
    def _assess_optimization_potential(self) -> Dict[str, Any]:
        """Assess potential for various optimizations"""
        potential = {
            'quantization': self._assess_quantization_potential(),
            'pruning': self._assess_pruning_potential(),
            'distillation': self._assess_distillation_potential(),
        }
        
        return potential
    
    def _assess_quantization_potential(self) -> Dict[str, Any]:
        """Assess quantization potential"""
        # Count quantization-friendly operations
        quantizable_ops = ['Conv', 'MatMul', 'Gemm', 'Add', 'Mul']
        
        total_ops = len(self.graph.node)
        quantizable_count = sum(
            1 for node in self.graph.node 
            if node.op_type in quantizable_ops
        )
        
        # Analyze weight distributions
        weight_stats = self._analyze_weight_distributions()
        
        return {
            'quantizable_percentage': (quantizable_count / total_ops * 100) if total_ops > 0 else 0,
            'quantizable_ops': quantizable_count,
            'total_ops': total_ops,
            'recommended_quantization': self._recommend_quantization_type(weight_stats),
            'weight_statistics': weight_stats,
        }
    
    def _analyze_weight_distributions(self) -> Dict[str, Any]:
        """Analyze weight distributions for quantization"""
        all_weights = []
        
        for init in self.initializers.values():
            tensor = numpy_helper.to_array(init)
            all_weights.append(tensor.flatten())
        
        if not all_weights:
            return {}
        
        all_weights = np.concatenate(all_weights)
        
        stats = {
            'mean': float(np.mean(all_weights)),
            'std': float(np.std(all_weights)),
            'min': float(np.min(all_weights)),
            'max': float(np.max(all_weights)),
            'median': float(np.median(all_weights)),
            'abs_max': float(np.max(np.abs(all_weights))),
        }
        
        return stats
    
    def _recommend_quantization_type(self, weight_stats: Dict) -> str:
        """Recommend quantization type based on weight statistics"""
        if not weight_stats:
            return 'Unknown'
        
        abs_max = weight_stats.get('abs_max', 0)
        std = weight_stats.get('std', 0)
        
        if abs_max < 10 and std < 1:
            return 'INT8 (Static or QAT recommended)'
        elif abs_max < 50:
            return 'INT8 (Dynamic or Static)'
        else:
            return 'FP16 or Mixed Precision (weights have large range)'
    
    def _assess_pruning_potential(self) -> Dict[str, Any]:
        """Assess pruning potential"""
        # Calculate sparsity in current weights
        total_params = 0
        near_zero_params = 0
        
        threshold = 1e-3  # Consider values below this as near-zero
        
        for init in self.initializers.values():
            tensor = numpy_helper.to_array(init)
            total_params += tensor.size
            near_zero_params += np.sum(np.abs(tensor) < threshold)
        
        current_sparsity = (near_zero_params / total_params * 100) if total_params > 0 else 0
        
        # Recommend pruning strategy
        pruning_recommendation = self._recommend_pruning_strategy(current_sparsity)
        
        return {
            'current_sparsity_percentage': current_sparsity,
            'prunable_parameters': total_params,
            'near_zero_parameters': near_zero_params,
            'recommended_pruning': pruning_recommendation,
        }
    
    def _recommend_pruning_strategy(self, current_sparsity: float) -> str:
        """Recommend pruning strategy based on current sparsity"""
        if current_sparsity > 50:
            return 'Already sparse - consider structured pruning for hardware efficiency'
        elif current_sparsity > 20:
            return 'Moderate sparsity - structured or unstructured pruning feasible'
        else:
            return 'Low sparsity - aggressive pruning possible (30-70% recommended)'
    
    def _assess_distillation_potential(self) -> Dict[str, Any]:
        """Assess knowledge distillation potential"""
        total_params = sum(
            numpy_helper.to_array(init).size 
            for init in self.initializers.values()
        )
        
        model_size_mb = get_model_size_mb(self.model_path)
        
        # Simple heuristic: larger models benefit more from distillation
        if model_size_mb > 100 or total_params > 50_000_000:
            recommendation = 'Highly recommended - large model can be distilled significantly'
        elif model_size_mb > 50 or total_params > 10_000_000:
            recommendation = 'Recommended - moderate compression possible'
        else:
            recommendation = 'Optional - small model, limited benefit'
        
        return {
            'model_size_mb': model_size_mb,
            'total_parameters': total_params,
            'recommendation': recommendation,
            'estimated_compression_ratio': '2-4x with accuracy preservation',
        }
    
    def print_summary(self, analysis: Optional[Dict] = None):
        """Print analysis summary"""
        if analysis is None:
            analysis = self.analyze()
        
        print("\n" + "=" * 80)
        print("ONNX MODEL ANALYSIS SUMMARY")
        print("=" * 80)
        
        # Model Info
        info = analysis['model_info']
        print(f"\n📊 Model Information:")
        print(f"  Name: {info['model_name']}")
        print(f"  Size: {info['model_size_mb']:.2f} MB")
        print(f"  Opset Version: {info['opset_version']}")
        
        # Inputs/Outputs
        print(f"\n🔌 Inputs:")
        for inp in info['inputs']:
            print(f"  - {inp['name']}: {inp['shape']} ({inp['dtype']})")
        
        print(f"\n🔌 Outputs:")
        for out in info['outputs']:
            print(f"  - {out['name']}: {out['shape']} ({out['dtype']})")
        
        # Architecture
        arch = analysis['architecture']
        print(f"\n🏗️  Architecture:")
        print(f"  Type: {arch['architecture_type']}")
        print(f"  Total Nodes: {arch['total_nodes']}")
        print(f"  Operator Types:")
        for op_type, count in sorted(arch['operator_types'].items(), 
                                     key=lambda x: x[1], reverse=True)[:10]:
            print(f"    - {op_type}: {count}")
        
        # Parameters
        params = analysis['parameters']
        print(f"\n🎯 Parameters:")
        print(f"  Total: {params['total_parameters']:,} ({params['total_parameters_millions']:.2f}M)")
        
        # Memory
        memory = analysis['memory']
        print(f"\n💾 Memory:")
        print(f"  Weights: {memory['weight_memory_mb']:.2f} MB")
        print(f"  Est. Activations: {memory['estimated_activation_memory_mb']:.2f} MB")
        print(f"  Peak Memory: {memory['peak_memory_mb']:.2f} MB")
        
        # Computation
        comp = analysis['computation']
        print(f"\n⚡ Computation:")
        print(f"  Total FLOPs: {comp['total_gflops']:.2f} GFLOPs")
        
        # Optimization Potential
        opt = analysis['optimization_potential']
        print(f"\n🔧 Optimization Potential:")
        
        print(f"\n  Quantization:")
        quant = opt['quantization']
        print(f"    - Quantizable ops: {quant['quantizable_percentage']:.1f}%")
        print(f"    - Recommendation: {quant['recommended_quantization']}")
        
        print(f"\n  Pruning:")
        prune = opt['pruning']
        print(f"    - Current sparsity: {prune['current_sparsity_percentage']:.2f}%")
        print(f"    - Recommendation: {prune['recommended_pruning']}")
        
        print(f"\n  Distillation:")
        distill = opt['distillation']
        print(f"    - Recommendation: {distill['recommendation']}")
        
        print("\n" + "=" * 80 + "\n")


# Convenience function
def analyze_onnx_model(model_path: Path) -> Dict[str, Any]:
    """Analyze ONNX model (convenience wrapper)"""
    analyzer = ONNXAnalyzer(model_path)
    return analyzer.analyze()
