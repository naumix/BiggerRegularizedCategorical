from typing import Callable
import jax.numpy as jnp
import flax.linen as nn
import distrax

def default_init(scale: float = jnp.sqrt(2)):
    return nn.initializers.orthogonal(scale)

class BronetBlock(nn.Module):
    hidden_dims: int
    activations: Callable[[jnp.ndarray], jnp.ndarray]

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        res = nn.Dense(self.hidden_dims, kernel_init=default_init())(x)
        res = nn.LayerNorm()(res)
        res = self.activations(res)
        res = nn.Dense(self.hidden_dims, kernel_init=default_init())(res)
        res = nn.LayerNorm()(res)
        return res + x

class BroNet(nn.Module):
    hidden_dims: int
    depth: int
    add_final_layer: bool = False
    output_nodes: int = 101
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = nn.Dense(self.hidden_dims, kernel_init=default_init())(x)
        x = nn.LayerNorm()(x)
        x = self.activations(x)
        for i in range(self.depth):
            x = BronetBlock(self.hidden_dims, self.activations)(x)
        if self.add_final_layer:
            x = nn.Dense(self.output_nodes, kernel_init=default_init())(x)
        return x

class TaskEmbedding(nn.Module): 
    num_tasks: int
    embedding_size: int
    
    def setup(self):
        self.embeddings = nn.Embed(self.num_tasks, self.embedding_size)
        
    def __call__(self, x: jnp.ndarray):
        emb = self.embeddings(x)
        norm = jnp.linalg.norm(emb, axis=-1, keepdims=True)
        emb = emb/norm
        return emb

class QValue(nn.Module):
    hidden_dims: int = 512
    depth: int = 2
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    output_nodes: int = 101
    
    def setup(self):
        self.critic = BroNet(hidden_dims=self.hidden_dims, depth=self.depth, activations=self.activations, add_final_layer=True, output_nodes=self.output_nodes)

    def __call__(self, inputs: jnp.ndarray):
        q_value = self.critic(inputs)
        return q_value    
    
class QValueEnsemble(nn.Module):
    ensemble_size: int = 2
    hidden_dims: int = 512
    depth: int = 2
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    output_nodes: int = 101
    
    def setup(self):
        VmapCritic = nn.vmap(QValue,
                             variable_axes={'params': 0},
                             split_rngs={'params': True},
                             in_axes=None,
                             out_axes=0,
                             axis_size=self.ensemble_size)
        self.q_value_ensemble = VmapCritic(hidden_dims=self.hidden_dims, depth=self.depth, activations=self.activations, output_nodes=self.output_nodes)

    def __call__(self, inputs: jnp.ndarray):
        q_values = self.q_value_ensemble(inputs)
        return q_values
    
class DoubleCriticTest(nn.Module):
    num_tasks: int
    embedding_size: int
    ensemble_size: int
    hidden_dims: int = 512
    depth: int = 2
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    output_nodes: int = 101
    multitask: bool = False
    
    def setup(self):
        if self.multitask:
            self.task_embedding = TaskEmbedding(self.num_tasks, self.embedding_size)
        self.q_value_ensemble = QValueEnsemble(
            ensemble_size=self.ensemble_size,
            hidden_dims=self.hidden_dims, 
            depth=self.depth,
            activations=self.activations,
            output_nodes=self.output_nodes,
        )

    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray, task_ids: jnp.ndarray, return_embeddings: bool = False):
        if self.multitask is False:
            inputs = jnp.concatenate((observations, actions), axis=-1)
        else:
            task_embedding = self.task_embedding(task_ids)
            if return_embeddings:
                return task_embedding
            inputs = jnp.concatenate((observations, actions, task_embedding), axis=-1)            
        q_values = self.q_value_ensemble(inputs)
        return q_values
    
class DoubleCritic(nn.Module):
    num_tasks: int
    embedding_size: int
    hidden_dims: int = 512
    depth: int = 2
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    output_nodes: int = 101
    multitask: bool = False
    
    def setup(self):
        if self.multitask:
            self.task_embedding = TaskEmbedding(self.num_tasks, self.embedding_size)
        self.critic1 = QValue(
            self.hidden_dims, 
            self.depth,
            activations=self.activations,
            output_nodes=self.output_nodes,
        )
        self.critic2 = QValue(
            self.hidden_dims, 
            self.depth,
            activations=self.activations,
            output_nodes=self.output_nodes,
        )

    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray, task_ids: jnp.ndarray, return_embeddings: bool = False):
        if self.multitask is False:
            inputs = jnp.concatenate((observations, actions), axis=-1)
        else:
            task_embedding = self.task_embedding(task_ids)
            if return_embeddings:
                return task_embedding
            inputs = jnp.concatenate((observations, actions, task_embedding), axis=-1)            
        q_value1 = self.critic1(inputs)
        q_value2 = self.critic2(inputs)
        return q_value1, q_value2

class NormalTanhPolicy(nn.Module):
    action_dim: int
    hidden_dims: int = 256
    depth: int = 1
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    log_std_scale: float = 1.0
    log_std_min: float =  -10.0
    log_std_max: float = 2.0

    @nn.compact
    def __call__(self, observations: jnp.ndarray, temperature: float = 1.0):
        outputs = BroNet(hidden_dims=self.hidden_dims, depth=self.depth, activations=self.activations, add_final_layer=False, output_nodes=None)(observations)
        means = nn.Dense(self.action_dim, kernel_init=default_init())(outputs)
        log_stds = nn.Dense(self.action_dim, kernel_init=default_init(self.log_std_scale))(outputs)
        log_stds = self.log_std_min + (self.log_std_max - self.log_std_min) * 0.5 * (1 + nn.tanh(log_stds))
        stds = jnp.exp(log_stds)
        stds = stds * temperature
        base_dist = distrax.MultivariateNormalDiag(loc=means, scale_diag=stds)
        tanh_dist = distrax.Transformed(base_dist, distrax.Block(distrax.Tanh(), 1))
        return tanh_dist
    
class Temperature(nn.Module):
    initial_temperature: float = 1.0
    
    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_temp = self.param('log_temp',init_fn=lambda key: jnp.full((), jnp.log(self.initial_temperature)))
        return jnp.exp(log_temp)