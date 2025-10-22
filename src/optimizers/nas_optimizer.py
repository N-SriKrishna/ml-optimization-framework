"""
Neural Architecture Search for Model Optimization
Automatically search for optimal pruning patterns and architectures
"""
from pathlib import Path
from typing import List, Tuple, Dict
import random
import numpy as np

import onnx
from onnx import numpy_helper

from src.utils.logger import logger


class NASOptimizer:
    """
    Neural Architecture Search for finding optimal model structures
    
    Uses evolutionary algorithms to search for:
    - Optimal pruning patterns
    - Layer width configurations
    - Skip connection patterns
    """
    
    def __init__(
        self,
        population_size: int = 20,
        generations: int = 10,
        mutation_rate: float = 0.2
    ):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        
        logger.info(f"Initialized NAS optimizer")
        logger.info(f"  Population: {population_size}")
        logger.info(f"  Generations: {generations}")
    
    def search(
        self,
        model_path: Path,
        fitness_fn: callable,
        target_sparsity: float = 0.5
    ) -> Dict:
        """
        Search for optimal model configuration
        
        Args:
            model_path: Path to ONNX model
            fitness_fn: Function that evaluates model (returns accuracy, latency, size)
            target_sparsity: Target sparsity level
        
        Returns:
            Best configuration found
        """
        logger.info(f"Starting NAS search...")
        logger.info(f"  Target sparsity: {target_sparsity*100:.0f}%")
        
        model = onnx.load(str(model_path))
        
        # Initialize population
        population = self._initialize_population(model, target_sparsity)
        
        best_individual = None
        best_fitness = -float('inf')
        
        for gen in range(self.generations):
            logger.info(f"\nGeneration {gen+1}/{self.generations}")
            
            # Evaluate population
            fitnesses = []
            for individual in population:
                # Apply configuration
                modified_model = self._apply_configuration(model, individual)
                
                # Evaluate
                metrics = fitness_fn(modified_model)
                fitness = self._compute_fitness(metrics, target_sparsity)
                fitnesses.append(fitness)
                
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_individual = individual
            
            logger.info(f"  Best fitness: {best_fitness:.4f}")
            
            # Selection
            population = self._selection(population, fitnesses)
            
            # Crossover
            population = self._crossover(population)
            
            # Mutation
            population = self._mutation(population)
        
        logger.info(f"\n✓ NAS search complete!")
        logger.info(f"  Best fitness: {best_fitness:.4f}")
        
        return {
            'configuration': best_individual,
            'fitness': best_fitness,
            'sparsity': self._calculate_sparsity(best_individual)
        }
    
    def _initialize_population(self, model, target_sparsity):
        """Initialize random population of configurations"""
        population = []
        
        # Get number of parameters per layer
        layer_sizes = []
        for init in model.graph.initializer:
            tensor = numpy_helper.to_array(init)
            if len(tensor.shape) >= 2:
                layer_sizes.append(tensor.size)
        
        # Generate random configurations
        for _ in range(self.population_size):
            # Each individual is a list of per-layer sparsity values
            individual = []
            for size in layer_sizes:
                # Random sparsity around target
                sparsity = target_sparsity + np.random.randn() * 0.1
                sparsity = np.clip(sparsity, 0.0, 0.9)
                individual.append(sparsity)
            
            population.append(individual)
        
        return population
    
    def _apply_configuration(self, model, configuration):
        """Apply pruning configuration to model"""
        # Clone model
        modified = onnx.ModelProto()
        modified.CopyFrom(model)
        
        # Apply sparsity per layer
        layer_idx = 0
        for init in modified.graph.initializer:
            tensor = numpy_helper.to_array(init)
            if len(tensor.shape) >= 2 and layer_idx < len(configuration):
                sparsity = configuration[layer_idx]
                
                # Apply pruning
                threshold = np.percentile(np.abs(tensor), sparsity * 100)
                mask = np.abs(tensor) >= threshold
                pruned = tensor * mask
                
                # Update
                new_init = numpy_helper.from_array(
                    pruned.astype(tensor.dtype),
                    name=init.name
                )
                init.CopyFrom(new_init)
                
                layer_idx += 1
        
        return modified
    
    def _compute_fitness(self, metrics, target_sparsity):
        """
        Compute fitness score
        
        Fitness = accuracy - λ₁|sparsity - target| - λ₂·latency
        """
        accuracy = metrics.get('accuracy', 0.0)
        latency = metrics.get('latency', 1.0)
        sparsity = metrics.get('sparsity', 0.0)
        
        # Penalties
        sparsity_penalty = abs(sparsity - target_sparsity) * 0.5
        latency_penalty = latency * 0.1
        
        fitness = accuracy - sparsity_penalty - latency_penalty
        
        return fitness
    
    def _selection(self, population, fitnesses):
        """Tournament selection"""
        selected = []
        
        for _ in range(self.population_size):
            # Tournament
            idx1, idx2 = random.sample(range(self.population_size), 2)
            if fitnesses[idx1] > fitnesses[idx2]:
                selected.append(population[idx1])
            else:
                selected.append(population[idx2])
        
        return selected
    
    def _crossover(self, population):
        """Crossover between individuals"""
        offspring = []
        
        for i in range(0, self.population_size, 2):
            parent1 = population[i]
            parent2 = population[min(i+1, self.population_size-1)]
            
            # Single-point crossover
            point = random.randint(0, len(parent1)-1)
            child1 = parent1[:point] + parent2[point:]
            child2 = parent2[:point] + parent1[point:]
            
            offspring.extend([child1, child2])
        
        return offspring[:self.population_size]
    
    def _mutation(self, population):
        """Mutate individuals"""
        for individual in population:
            if random.random() < self.mutation_rate:
                # Mutate random gene
                idx = random.randint(0, len(individual)-1)
                individual[idx] += np.random.randn() * 0.05
                individual[idx] = np.clip(individual[idx], 0.0, 0.9)
        
        return population
    
    def _calculate_sparsity(self, configuration):
        """Calculate average sparsity"""
        return np.mean(configuration)
