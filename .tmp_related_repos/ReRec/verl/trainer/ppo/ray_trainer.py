# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import uuid
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Dict, Optional, Type, List, Any, Tuple

import numpy as np
import ray
import torch
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.metric import (
    reduce_metrics,
)
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.rollout.async_server import AsyncLLMServerManager
from verl.utils.dataset.rl_dataset import DataProtoDataset

WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """
    RAAE = "raae"

@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0) for node, node_info in node_available_resources.items()}

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])
        if total_available_gpus < total_required_gpus:
            raise ValueError(f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}")

        # check each resource pool can be satisfied, O(#resource_pools * #nodes)
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes}" + "cannot be satisfied in this ray cluster")


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl", multi_turn=False):
    responses = data.batch["responses"]
    response_length = responses.size(1)
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    if multi_turn:
        loss_mask = data.batch["loss_mask"]
        response_mask = loss_mask[:, -response_length:]
    else:
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty)  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1, multi_turn=False, norm_adv_by_std_in_grpo=True):
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch:
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    
    if adv_estimator == AdvantageEstimator.RAAE:
        advantages, returns = core_algos.compute_raae_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        raise NotImplementedError
    return data


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    if name not in timing_raw:
        timing_raw[name] = 0
    timing_raw[name] += timer.last


@dataclass
class SampleData:
    """Store all relevant information for a single sample"""
    original_data: Any  # Original batch data containing multiple samples with batch, non_tensor_batch, meta_info three attributes
    extra_infos: List[Dict[str, Any]] = field(default_factory=list)  # Additional information for all rewards of this sample

@dataclass
class EpochCollectedData:
    """Store all sample data and related information collected in an epoch"""
    samples: Dict[int, SampleData]  # key is the original index of the sample, value is all information of this sample
    n_repeats: int  # Number of times each sample is repeated
    
    @classmethod
    def create_empty(cls, n_repeats):
        """Create an empty data collector"""
        return cls(samples={}, n_repeats=n_repeats)
    
    def append_batch(self, batch_idx_start: int, sample_data, is_original: bool = True, rewards=None, extra_infos=None):
        """
        Add a batch of data
        
        Args:
            batch_idx_start: Original index of the first sample in this batch
            sample_data: Original data in the batch (DataProto object or None)
            is_original: Whether it is original data (not repeated)
            rewards: List of rewards for all responses in the batch
            extra_infos: List of additional information for all responses in the batch
        """
        if is_original:
            # Process original batch data
            assert sample_data is not None, "sample_data cannot be None when is_original=True"
            assert isinstance(sample_data, DataProto), "sample_data must be DataProto when is_original=True"
            
            # Use batch_size attribute to get batch size
            batch_size = len(sample_data)  # Use __len__ method to get batch size
            
            # Check index continuity
            if len(self.samples) > 0:
                max_existing_idx = max(self.samples.keys())
                assert batch_idx_start == max_existing_idx + 1, \
                    f"Non-continuous batch index: expected {max_existing_idx + 1}, got {batch_idx_start}"
            
            # Create a DataProto object for each sample
            for i in range(batch_size):
                orig_idx = batch_idx_start + i
                if orig_idx not in self.samples:
                    # Use __getitem__ method to get single sample
                    single_sample = sample_data[i]  # This will return DataProtoItem
                    self.samples[orig_idx] = SampleData(original_data=single_sample)
                else:
                    assert False, f"Sample index {orig_idx} already exists"
        else:
            # Process generated data
            assert extra_infos is not None, \
                "At least one of rewards or extra_infos must be provided when is_original=False"
            
            if extra_infos is not None:
                # Assume the length of the first list in extra_infos represents the number of samples
                first_key = next(iter(extra_infos.keys()))
                batch_size = len(extra_infos[first_key]) // self.n_repeats
                # Ensure each extra_info has the same length and is an integer multiple of n_repeats
                for k, v in extra_infos.items():
                    assert len(v) == batch_size * self.n_repeats, \
                        f"Length mismatch in extra_infos: key {k} has length {len(v)}, " \
                        f"expected {batch_size * self.n_repeats}"
            
            for i in range(batch_size):
                orig_idx = batch_idx_start + i
                # Ensure the corresponding original sample exists
                assert orig_idx in self.samples, f"Original sample with index {orig_idx} not found"
                
                if extra_infos is not None:
                    # Collect additional information for all responses of this sample
                    sample_extra_infos = []
                    for j in range(self.n_repeats):
                        response_idx = i * self.n_repeats + j
                        info = {k: v[response_idx] for k, v in extra_infos.items()}
                        sample_extra_infos.append(info)
                    assert len(sample_extra_infos) == self.n_repeats, \
                        f"Expected {self.n_repeats} extra_infos for sample {orig_idx}, got {len(sample_extra_infos)}"
                    self.samples[orig_idx].extra_infos.extend(sample_extra_infos)

    def validate_data(self):
        """Validate that the collected data is complete and correct"""
        # Check index continuity
        indices = sorted(self.samples.keys())
        assert len(indices) > 0, "No samples collected"
        assert indices[0] == 0, f"First sample index should be 0, got {indices[0]}"
        assert indices[-1] == len(indices) - 1, \
            f"Missing samples: expected {len(indices)} samples, got max index {indices[-1]}"
        
        # Check the number of rewards and extra_infos for each sample
        for idx, sample in self.samples.items():
            if len(sample.extra_infos) > 0:
                assert len(sample.extra_infos) == self.n_repeats, \
                    f"Sample {idx} has {len(sample.extra_infos)} extra_infos, expected {self.n_repeats}"


class CurriculumSample:
    """Sample information for curriculum learning"""
    def __init__(self, original_data: Any, difficulty: float, uncertainty: float):
        self.original_data = original_data
        self.difficulty = difficulty
        self.uncertainty = uncertainty
    
    @staticmethod
    def compute_cv(values: List[float]) -> float:
        """Compute coefficient of variation (CV)"""
        if not values:
            return 0.0
        values = np.array(values)
        return np.std(values) / (np.mean(values) + 1e-8)  # Add small value to avoid division by zero

class CurriculumLearningManager:
    """Manage sample selection and ordering for curriculum learning"""
    def __init__(self, config):
        self.config = config
        self.easy_samples: Dict[int, CurriculumSample] = {}  # Store easy samples
        
    def process_samples(self, epoch_data: EpochCollectedData) -> Tuple[List[CurriculumSample], List[int]]:
        """Process samples, return training samples and easy sample indices"""
        curriculum_config = self.config.trainer.curriculum_learning
        difficulty_metric = curriculum_config.difficulty_metric
        uncertainty_metric = curriculum_config.uncertainty_metric
        
        # Collect information for all samples
        all_samples: Dict[int, CurriculumSample] = {}
        
        for idx, sample_data in epoch_data.samples.items():
            # Get metric values for all responses
            difficulty_values = []
            uncertainty_values = []
            
            for extra_info in sample_data.extra_infos:
                if difficulty_metric in extra_info:
                    # difficulty = 1 - metric
                    difficulty_values.append(1 - extra_info[difficulty_metric])
                if uncertainty_metric in extra_info:
                    uncertainty_values.append(extra_info[uncertainty_metric])
            
            # Calculate difficulty (take average) and uncertainty (compute coefficient of variation)
            avg_difficulty = np.mean(difficulty_values) if difficulty_values else 1.0
            uncertainty = CurriculumSample.compute_cv(uncertainty_values)
            
            all_samples[idx] = CurriculumSample(
                original_data=sample_data.original_data,
                difficulty=avg_difficulty,
                uncertainty=uncertainty
            )
        
        # Divide samples according to thresholds
        difficulty_threshold = curriculum_config.difficulty_threshold
        uncertainty_threshold = curriculum_config.uncertainty_threshold
        
        train_samples = []
        easy_indices = []
        
        for idx, sample in all_samples.items():
            # if sample.uncertainty < uncertainty_threshold and sample.difficulty < difficulty_threshold:
            if sample.difficulty < difficulty_threshold:
                # Easy samples: both uncertainty and difficulty are below threshold
                easy_indices.append(idx)
                self.easy_samples[idx] = sample
            else:
                # Training samples: all other cases
                train_samples.append(sample)
        
        # Sort training samples: first by uncertainty descending, then by difficulty ascending
        # train_samples.sort(key=lambda x: (-x.uncertainty, x.difficulty))
        train_samples.sort(key=lambda x: x.difficulty)
        
        return train_samples, easy_indices

class RayPPOTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name="cuda",
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name
        self.validation_generations_logger = ValidationGenerationsLogger()

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
            AdvantageEstimator.GRPO,
            AdvantageEstimator.GRPO_PASSK,
            AdvantageEstimator.REINFORCE_PLUS_PLUS,
            AdvantageEstimator.REMAX,
            AdvantageEstimator.RLOO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
            AdvantageEstimator.RAAE,
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError

        self._validate_config()
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

        # Initialize best checkpoints tracker
        self.best_checkpoints = []  # [(global_step, metric_value), ...]
        self.val_metric_key = self.config.trainer.get("val_metric_key", "val-core/all/reward/mean@1")  # Metric for judging model quality
        self.val_metric_mode = self.config.trainer.get("val_metric_mode", "max")  # 'max' or 'min'
        self.last_val_metrics = None  # Add this line for initialization

        from verl.utils.tracking import Tracking

        self.logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % n_gpus == 0, f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            settings = {
                "actor_rollout_ref.actor": "micro_batch_size",
                "critic": "micro_batch_size",
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove '{name}.{param}' because only '*_{param_per_gpu}'" + "is supported (the former is deprecated).")

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.actor.ppo_micro_batch_size,
                config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                "actor_rollout_ref.actor",
            )

            if self.use_reference_policy:
                # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
                check_mutually_exclusive(
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                    "actor_rollout_ref.ref",
                )

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.rollout",
            )

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu, "critic")

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu, "reward_model")

        # Actor
        # check if train_batch_size is larger than ppo_mini_batch_size
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.actor_rollout_ref.actor.ppo_mini_batch_size
            sp_size = config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        assert config.actor_rollout_ref.actor.loss_agg_mode in [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        ], f"Invalid loss_agg_mode: {config.actor_rollout_ref.actor.loss_agg_mode}"

        if config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
            print("NOTICE: You have both enabled in-reward kl and kl loss.")

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size
            sp_size = config.critic.get("ulysses_sequence_parallel_size", 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == "fsdp" and (config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1) > 1 or config.actor_rollout_ref.ref.get("ulysses_sequence_parallel_size", 1) > 1):
            assert config.actor_rollout_ref.model.use_remove_padding, "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."

        if self.use_critic and config.critic.strategy == "fsdp":
            if config.critic.get("ulysses_sequence_parallel_size", 1) > 1:
                assert config.critic.model.use_remove_padding, "When using sequence parallelism for critic, you must enable `use_remove_padding`."

        if config.data.get("val_batch_size", None) is not None:
            print("WARNING: val_batch_size is deprecated." + " Validation datasets are sent to inference engines as a whole batch," + " which will schedule the memory themselves.")

        # check eval config
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, "validation gen temperature should be greater than 0 when enabling do_sample"

        # check multi_turn with tool config
        if config.actor_rollout_ref.rollout.multi_turn.enable:
            assert config.actor_rollout_ref.rollout.multi_turn.tool_config_path is not None, "tool_config_path must be set when enabling multi_turn with tool, due to no role-playing support"
            assert config.algorithm.adv_estimator in [AdvantageEstimator.GRPO], "only GRPO is tested for multi-turn with tool"

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(self.config.data.train_files, self.config.data, self.tokenizer, self.processor)
        if val_dataset is None:
            val_dataset = create_rl_dataset(self.config.data.val_files, self.config.data, self.tokenizer, self.processor)
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: {len(self.val_dataloader)}")

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        with open(filename, "w") as f:
            for i in range(n):
                entry = {k: v[i] for k, v in base_data.items()}
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"Dumped generations to {filename}")

    def _maybe_log_val_generations(self, inputs, outputs, labels, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, labels, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _log_val_metrics(self, val_metrics: Dict[str, Any]) -> None:
        """Log validation metrics"""
        # val_metrics structure: 
        # {
        #   True: {difficult_mode: {category: [metrics_dict, ...], ...}, ...},
        #   False: {category: [metrics_dict, ...], ...}
        # }
        assert len(val_metrics) > 0, "No category metrics to log"
        
        for with_history, data_dict in val_metrics.items():
            if with_history:
                # When with_history is True, need to handle nested structure of difficult_mode
                all_with_history_scores = []
                for difficult_mode, category_dict in data_dict.items():
                    all_difficult_mode_scores = []
                    for category, metrics_list in category_dict.items():
                        # Calculate scores for all categories
                        scores = [d["ndcg@k"] for d in metrics_list if "ndcg@k" in d]
                        if scores:
                            avg_score = np.mean(scores)
                            self.logger.log(
                                data={f"val/with_history_{with_history}/{difficult_mode}/{category}/ndcg@k": avg_score},
                                step=self.global_steps
                            )
                            all_difficult_mode_scores.extend(scores)
                            all_with_history_scores.extend(scores)
                    
                    # Calculate average score for this difficult_mode
                    if all_difficult_mode_scores:
                        avg_difficult_mode_score = np.mean(all_difficult_mode_scores)
                        self.logger.log(
                            data={f"val/with_history_{with_history}/{difficult_mode}/ndcg@k": avg_difficult_mode_score},
                            step=self.global_steps
                        )
                
                # Calculate total average score for with_history=True group
                if all_with_history_scores:
                    avg_with_history_score = np.mean(all_with_history_scores)
                    self.logger.log(
                        data={f"val/with_history_{with_history}/ndcg@k": avg_with_history_score},
                        step=self.global_steps
                    )
            else:
                # When with_history is False, process directly by category (maintain original logic)
                all_scores = []
                for category, metrics_list in data_dict.items():
                    # Calculate scores for all categories
                    scores = [d["ndcg@k"] for d in metrics_list if "ndcg@k" in d]
                    if scores:
                        avg_score = np.mean(scores)
                        self.logger.log(
                            data={f"val/with_history_{with_history}/{category}/ndcg@k": avg_score},
                            step=self.global_steps
                        )
                        all_scores.extend(scores)
                
                # Calculate average score for with_history=False group
                if all_scores:
                    avg_with_history_score = np.mean(all_scores)
                    self.logger.log(
                        data={f"val/with_history_{with_history}/ndcg@k": avg_with_history_score},
                        step=self.global_steps
                    )

    def _validate(self):
        reward_tensor_lst = []
        sample_inputs, sample_outputs, sample_labels, sample_scores, sample_metrics = [], [], [], [], []
        reward_metrics_lst = defaultdict(list)
        # Use more flexible data structure
        with_history_category_metrics = {}
        
        for batch_dict in self.val_dataloader:
            test_batch = DataProto.from_single_dict(batch_dict)
            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            
            test_gen_batch = test_batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids"],
            )

            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            test_gen_batch, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            
            test_output_gen_batch = self.actor_rollout_wg.generate_sequences(test_gen_batch)
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch, pad_size=pad_size)

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)
            
            reward_model_data = test_batch.non_tensor_batch["reward_model"]
            sample_labels.extend([data["ground_truth"] for data in reward_model_data])
                
            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            reward_metrics = result["reward_extra_info"]
            max_response_length = test_batch.batch["responses"].shape[-1]
            response_mask = test_batch.batch["attention_mask"][:, -max_response_length:].bool()
            # Set invalid positions to -inf, compatible with raae and grpo
            reward_tensor.masked_fill_(~response_mask, float('-inf'))
            reward_tensor_lst.append(reward_tensor)
            
            # Take maximum value of valid parts
            scores = reward_tensor.max(-1).values.cpu().tolist()

            # Store scores
            # if self.config.reward_model.reward_manager == "raae":
            #     # value is score in reward_metrics
            #     scores = [reward_metrics["score"][i] for i in range(len(reward_metrics["score"]))]
            # else:
            #     scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            for key, value in reward_metrics.items():
                reward_metrics_lst[key].extend(value)

    
            # Process category-based metrics
            if "extra_info" in test_batch.non_tensor_batch:
                for i, extra_info in enumerate(test_batch.non_tensor_batch["extra_info"]):
                    category = extra_info.get("category", "unknown")
                    assert "with_history" in extra_info, f"extra_info must contain with_history field"
                    with_history = extra_info.get("with_history", False)
                    metrics_dict = {k: v[i] for k, v in reward_metrics.items()}
                    
                    # Initialize data structure
                    if with_history not in with_history_category_metrics:
                        with_history_category_metrics[with_history] = {}
                    
                    if with_history:
                        # When with_history is True, further classify by difficult_mode
                        difficult_mode = extra_info.get("difficulty_mode", "unknown")
                        if difficult_mode not in with_history_category_metrics[with_history]:
                            with_history_category_metrics[with_history][difficult_mode] = {}
                        if category not in with_history_category_metrics[with_history][difficult_mode]:
                            with_history_category_metrics[with_history][difficult_mode][category] = []
                        with_history_category_metrics[with_history][difficult_mode][category].append(metrics_dict)
                    else:
                        # When with_history is False, classify directly by category
                        if category not in with_history_category_metrics[with_history]:
                            with_history_category_metrics[with_history][category] = []
                        with_history_category_metrics[with_history][category].append(metrics_dict)
        
        # Log output
        self._log_val_metrics(with_history_category_metrics)
        reward_metrics = [str(reward_dict) for reward_dict in reward_metrics]
        sample_metrics.extend(reward_metrics)
        self._maybe_log_val_generations(sample_inputs, sample_outputs, sample_labels, sample_scores)
        reward_score = torch.cat(reward_tensor_lst, dim=0).max(-1).values.mean().item()
        val_reward_metrics = {f"val/{key}": value for key, value in reduce_metrics(reward_metrics_lst).items()}
        return {"val/reward_score": reward_score, **val_reward_metrics}

    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy], config=self.config.actor_rollout_ref, role="ref")
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls, device_name=self.device_name, **wg_kwargs)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            self.async_rollout_mode = True
            self.async_rollout_manager = AsyncLLMServerManager(
                config=self.config.actor_rollout_ref,
                worker_group=self.actor_rollout_wg,
            )


    def _update_best_checkpoints(self, global_step, metric_value):
        """Update best checkpoint list"""
        max_keep = self.config.trainer.get("max_actor_ckpt_to_keep", 1)
        
        # Add current checkpoint to list
        self.best_checkpoints.append((global_step, metric_value))
        
        # Sort by metric value (considering metric_mode)
        self.best_checkpoints.sort(key=lambda x: x[1], reverse=(self.val_metric_mode == "max"))
        
        # Keep only the best max_keep ones
        self.best_checkpoints = self.best_checkpoints[:max_keep]
        
        return global_step in [x[0] for x in self.best_checkpoints]

    def _remove_old_checkpoint(self, global_step):
        """Remove old checkpoints that are not in the best list"""
        checkpoint_dir = os.path.join(self.config.trainer.default_local_dir, f"global_step_{global_step}")
        if os.path.exists(checkpoint_dir):
            import shutil
            shutil.rmtree(checkpoint_dir)
            print(f"Removed old checkpoint: {checkpoint_dir}")

    def _save_checkpoint(self):
        """Save checkpoint only when validation performance improves"""
        if not hasattr(self, 'last_val_metrics') or not self.last_val_metrics:
            return
        
        current_metric = self.last_val_metrics.get(self.val_metric_key)
        if current_metric is None:
            print(f"Warning: Validation metric {self.val_metric_key} not found")
            return
        
        # Determine whether to save current checkpoint
        should_save = self._update_best_checkpoints(self.global_steps, current_metric)
        
        if should_save:
            print(f"Found better checkpoint, metric {self.val_metric_key}: {current_metric}")
            # Save current checkpoint
            local_global_step_folder = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")
            actor_local_path = os.path.join(local_global_step_folder, "actor")
            actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
                self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor"
            )
            
            self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path, self.global_steps)
            
            if self.use_critic:
                critic_local_path = os.path.join(local_global_step_folder, "critic")
                critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
                    self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic"
                )
                self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path, self.global_steps)
            
            # Save dataloader state
            dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
            dataloader_state_dict = self.train_dataloader.state_dict()
            torch.save(dataloader_state_dict, dataloader_local_path)
            
            # Update latest checkpointed iteration
            local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt")
            with open(local_latest_checkpointed_iteration, "w") as f:
                f.write(str(self.global_steps))
            
            print(f"Saved current checkpoint to {local_global_step_folder}")
        
        # Clean up old checkpoints not in the best list
        all_checkpoints = [d for d in os.listdir(self.config.trainer.default_local_dir) if d.startswith("global_step_")]
        for ckpt in all_checkpoints:
            step = int(ckpt.split("_")[-1])
            if step not in [x[0] for x in self.best_checkpoints]:
                self._remove_old_checkpoint(step)

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            # load from hdfs
            if self.config.trainer.default_hdfs_dir is not None:
                raise NotImplementedError("load from hdfs is not implemented yet")
            else:
                checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
                if not os.path.isabs(checkpoint_folder):
                    working_dir = os.getcwd()
                    checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
                global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        # load actor
        self.actor_rollout_wg.load_checkpoint(actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst, k_partitions=world_size, equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def _reorder_and_rebatch_samples(self, epoch_data: EpochCollectedData):
        """
        Reorder and rebatch samples in epoch based on reward and additional information
        """
        # Ensure curriculum_learning related configuration exists
        assert hasattr(self.config.trainer, "curriculum_learning"), "Missing curriculum_learning config"
        assert self.config.trainer.curriculum_learning.enable, "Curriculum learning is not enabled"
        
        # If called for the first time, initialize curriculum_learning_manager
        if not hasattr(self, 'curriculum_learning_manager'):
            self.curriculum_learning_manager = CurriculumLearningManager(self.config)
        
        # Process samples
        train_samples, easy_indices = self.curriculum_learning_manager.process_samples(epoch_data)
        
        print(f"Curriculum Learning: {len(train_samples)} samples for training, "
              f"{len(easy_indices)} samples marked as easy")
        
        # Get data file path from original dataset
        # Check if original_data format meets requirements
        first_sample = train_samples[0].original_data if train_samples else None
        # if first_sample is not None:
        #     print(f"Sample data type: {type(first_sample)}")
        #     print(f"Sample data keys: {first_sample.keys() if isinstance(first_sample, dict) else 'not a dict'}")
        
        # Build data that meets requirements
        train_data = [sample.original_data for sample in train_samples]
        
        # Create new dataset
        reordered_dataset = DataProtoDataset(
            samples=train_data,  # Directly pass DataProtoItem list
            tokenizer=self.tokenizer,
            config=self.config.data,
            processor=self.processor
        )
        
        # Record curriculum learning statistics
        curriculum_stats = {
            "curriculum/last_epoch_samples": len(epoch_data.samples),
            "curriculum/train_samples": len(train_samples),
            "curriculum/easy_samples": len(easy_indices),
            "curriculum/total_samples": len(train_samples) + len(easy_indices),
            "curriculum/easy_ratio": len(easy_indices) / (len(train_samples) + len(easy_indices))
        }
        print("Curriculum Learning Stats:", curriculum_stats)
        
        # Save easy sample information for possible re-evaluation later
        self.curriculum_learning_manager.easy_samples = {
            idx: epoch_data.samples[idx] for idx in easy_indices
        }
        
        return reordered_dataset

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        logger = self.logger

        

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        print("val before train: ", self.config.trainer.get("val_before_train", True))
        assert self.val_reward_fn is not None 
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            print("Start validation before training")
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return
            print("End validation, start training!")

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        for epoch in range(self.config.trainer.total_epochs):
            n_repeats = self.config.actor_rollout_ref.rollout.n
            epoch_data = EpochCollectedData.create_empty(n_repeats=n_repeats)
            
            batch_idx_start = 0  # Track the starting index of current batch
            for batch_dict in self.train_dataloader:
                # print(f"batch size: {batch_dict['input_ids'].shape[0]}")
                # First create DataProto object
                # Determine if it's dict or DataProto
                if isinstance(batch_dict, dict):
                    batch: DataProto = DataProto.from_single_dict(batch_dict)
                else:
                    batch = batch_dict
            
                # Use DataProto object as sample_data
                epoch_data.append_batch(
                    batch_idx_start=batch_idx_start,
                    sample_data=batch.copy(),  # Pass DataProto object
                    is_original=True
                )
                
                metrics = {}
                timing_raw = {}

                # pop those keys for generation
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
                if "multi_modal_inputs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.extend(["multi_modal_data", "multi_modal_inputs"])
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer("step", timing_raw):
                    # generate a batch
                    print("[INFO] Start generating sequences")
                    with _timer("gen", timing_raw):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        else:
                            self.async_rollout_manager.wake_up()
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                            self.async_rollout_manager.sleep()

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with _timer("gen_max", timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    batch.batch["response_mask"] = compute_response_mask(batch)
                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    print("[INFO] Start computing reward score")
                    with _timer("reward", timing_raw):
                        # compute reward model score
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        # whether the reward function is dual graph enhanced
                        batch.meta_info['dual_graph_enhanced'] = self.config.custom_reward_function.dual_graph_enhanced
                        print(f"[INFO] Dual graph enhanced: {batch.meta_info['dual_graph_enhanced']}")
                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)
                        
                        # log min,max,mean values of reward_extra_infos_dict
                        vals = ['min', 'max', 'mean']
                        reward_metrics = {}
                        for k, v in reward_extra_infos_dict.items():
                            for val in vals:
                                reward_metrics[f"reward/{k}/{val}"] = v
                                # print("type of reward_metrics[f'reward/{k}/{val}']", type(reward_metrics[f'reward/{k}_{val}']))
                                # print("shape of reward_metrics[f'reward/{k}/{val}']", reward_metrics[f'reward/{k}_{val}'].shape)
                        metrics.update(reduce_metrics(reward_metrics))
                        print(f"Reward metrics: {metrics}")

                        # Update rewards and extra_infos for each sample in this batch
                        epoch_data.append_batch(
                            batch_idx_start=batch_idx_start,
                            sample_data=None,
                            is_original=False,  # Mark this as generated data
                            extra_infos=reward_extra_infos_dict
                        )

                    # recompute old_log_probs
                    print("[INFO] Start computing old log prob")
                    with _timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_loss = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy_loss": entropy_loss.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer("ref", timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    
                    with _timer("adv", timing_raw):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor
                        

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})



                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # compute advantages, executed on the driver process

                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)  # GRPO adv normalization factor

                        print("[INFO] Start computing advantages")
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                        )
            

                    # update critic
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        print("[INFO] Start updating actor")
                        with _timer("update_actor", timing_raw):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with _timer("dump_rollout_generations", timing_raw):
                            # print(batch.batch.keys())
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            # Get valid mask
                            max_response_length = batch.batch["responses"].shape[-1]
                            response_mask = batch.batch["attention_mask"][:, -max_response_length:].bool()
                            
                            # Set invalid positions to -inf, compatible with raae and grpo
                            scores_tensor = batch.batch["token_level_scores"].clone()
                            scores_tensor.masked_fill_(~response_mask, float('-inf'))
                            
                            # Take maximum value of valid parts
                            scores = scores_tensor.max(-1).values.cpu().tolist()
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                            self.last_val_metrics = val_metrics  # First assignment here
                        metrics.update(val_metrics)

                    # Try to save checkpoint after validation
                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                        "training/samples":len(self.train_dataloader) * self.config.data.train_batch_size
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1
                batch_idx_start += batch_dict["input_ids"].shape[0]
            
            # After epoch ends, reorder and rebatch samples
            if epoch < self.config.trainer.total_epochs - 1 and self.config.trainer.curriculum_learning.enable:  # Not the last epoch
                print(f"Epoch {epoch} finished. Reordering and rebatching samples...")
                
                # Validate collected data
                epoch_data.validate_data()
                
                # Use integrated data structure to call reordering function
                reordered_dataset = self._reorder_and_rebatch_samples(epoch_data)
                
                # Recreate dataloader
                from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn
                collate_fn = default_collate_fn
                if len(reordered_dataset) >= self.config.data.train_batch_size: 
                    self.train_dataloader = StatefulDataLoader(
                        dataset=reordered_dataset,
                        batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
                        num_workers=self.config.data.get("dataloader_num_workers", 8),
                        drop_last=True,
                        collate_fn=collate_fn,
                        shuffle=False,  # Already reordered, no need to shuffle again
                    )
                    print(f"New dataloader created with {len(self.train_dataloader)} batches")
                else:
                    print("Less than 1 batch, no need to create new dataloader")
                    from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn
                    collate_fn = default_collate_fn
                    
                    # Use original dataset to recreate dataloader
                    self.train_dataloader = StatefulDataLoader(
                        dataset=self.train_dataset,  # Use original training dataset
                        batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
                        num_workers=self.config.data.get("dataloader_num_workers", 8),
                        drop_last=True,
                        collate_fn=collate_fn,
                        shuffle=False,
                    )
                    

