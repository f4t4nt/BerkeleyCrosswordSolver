from typing import Callable, Optional, Sequence, Tuple
import copy

import torch
from torch import nn
import numpy as np

# import cs285.infrastructure.pytorch_util as ptu
import pytorch_util as ptu


class SoftActorCritic(nn.Module):
    def __init__(
        self,
        make_actor: Callable[[Tuple[int, ...], int], nn.Module],
        make_actor_optimizer: Callable[[torch.nn.ParameterList], torch.optim.Optimizer],
        make_actor_schedule: Callable[
            [torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler
        ],
        make_critic: Callable[[Tuple[int, ...], int], nn.Module],
        make_critic_optimizer: Callable[
            [torch.nn.ParameterList], torch.optim.Optimizer
        ],
        make_critic_schedule: Callable[
            [torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler
        ],
        discount: float,
        target_update_period: Optional[int] = None,
        soft_target_update_rate: Optional[float] = None,
        # Actor-critic configuration
        actor_gradient_type: str = "reinforce",  # One of "reinforce" or "reparametrize"
        num_actor_samples: int = 1,
        num_critic_updates: int = 1,
        # Settings for multiple critics
        num_critic_networks: int = 1,
        target_critic_backup_type: str = "mean",  # One of "doubleq", "min", "redq", or "mean"
        # Soft actor-critic
        use_entropy_bonus: bool = False,
        temperature: float = 0.0,
        backup_entropy: bool = True,
    ):
        super().__init__()

        assert target_critic_backup_type in [
            "doubleq",
            "min",
            "mean",
            "redq",
        ], f"{target_critic_backup_type} is not a valid target critic backup type"

        assert actor_gradient_type in [
            "reinforce",
            "reparametrize",
        ], f"{actor_gradient_type} is not a valid type of actor gradient update"

        assert (
            target_update_period is not None or soft_target_update_rate is not None
        ), "Must specify either target_update_period or soft_target_update_rate"

        self.actor = ... #load gpt model
        self.actor_optimizer = make_actor_optimizer(self.actor.parameters())
        self.actor_lr_scheduler = make_actor_schedule(self.actor_optimizer)

        self.critics = nn.ModuleList(
            [
                ... #load gpt model
            ]
        )

        self.critic_optimizer = make_critic_optimizer(self.critics.parameters())
        self.critic_lr_scheduler = make_critic_schedule(self.critic_optimizer)
        self.target_critics = nn.ModuleList(
            [
                ... #load gpt model
            ]
        )
        self.update_target_critic()
        self.discount = discount
        self.target_update_period = target_update_period
        self.target_critic_backup_type = target_critic_backup_type
        self.num_critic_networks = num_critic_networks
        self.use_entropy_bonus = use_entropy_bonus
        self.temperature = temperature
        self.actor_gradient_type = actor_gradient_type
        self.num_actor_samples = num_actor_samples
        self.num_critic_updates = num_critic_updates
        self.soft_target_update_rate = soft_target_update_rate
        self.backup_entropy = backup_entropy
        self.max_length = 60
        self.critic_loss = nn.MSELoss()

        self.update_target_critic()

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """
        Compute the action for a given observation.
        """
        with torch.no_grad():
            observation = ptu.from_numpy(observation)[None]

            action, _ = self.get_actions(observation)
            return ptu.to_numpy(action).squeeze(0)

    def critic(self, state_action: torch.Tensor) -> torch.Tensor:
        """
        Compute the (ensembled) Q-values for the given state-action pair.
        """
        return torch.stack([critic(state_action) for critic in self.critics], dim=0)

    def target_critic(self, state_action: torch.Tensor) -> torch.Tensor:
        """
        Compute the (ensembled) target Q-values for the given state-action pair.
        """
        return torch.stack(
            [critic(state_action) for critic in self.target_critics], dim=0
        )

    def q_backup_strategy(self, next_qs: torch.Tensor) -> torch.Tensor:
        """
        Handle Q-values from multiple different target critic networks to produce target values.

        For example:
         - for "vanilla", we can just leave the Q-values as-is (we only have one critic).
         - for double-Q, swap the critics' predictions (so each uses the other as the target).
         - for clip-Q, clip to the minimum of the two critics' predictions.

        Parameters:
            next_qs (torch.Tensor): Q-values of shape (num_critics, batch_size). 
                Leading dimension corresponds to target values FROM the different critics.
        Returns:
            torch.Tensor: Target values of shape (num_critics, batch_size). 
                Leading dimension corresponds to target values FOR the different critics.
        """

        assert (
            next_qs.ndim == 2
        ), f"next_qs should have shape (num_critics, batch_size) but got {next_qs.shape}"
        num_critic_networks, batch_size = next_qs.shape
        assert num_critic_networks == self.num_critic_networks

        # TODO(student): Implement the different backup strategies.
        if self.target_critic_backup_type == "doubleq":
            # raise NotImplementedError
            next_qs = next_qs[[1, 0]] # FLAG
        elif self.target_critic_backup_type == "min":
            # raise NotImplementedError
            next_qs, _ = next_qs.min(dim=0)
        else:
            # Default, we don't need to do anything.
            pass


        # If our backup strategy removed a dimension, add it back in explicitly
        # (assume the target for each critic will be the same)
        if next_qs.shape == (batch_size,):
            next_qs = next_qs[None].expand((self.num_critic_networks, batch_size)).contiguous()

        assert next_qs.shape == (
            self.num_critic_networks,
            batch_size,
        ), next_qs.shape
        return next_qs

    def update_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
    ):
        """
        Update the critic networks by computing target values and minimizing Bellman error.
        """
        (batch_size,) = reward.shape

        # Compute target values
        # Important: we don't need gradients for target values!
        with torch.no_grad():
            # TODO(student)
            # Sample from the actor
            # next_action_distribution: torch.distributions.Distribution = ...
            # next_action = ...
            next_action, next_action_log_probs = self.get_actions(next_obs)

            # Compute the next Q-values for the sampled actions
            # next_qs = ...
            next_qs = self.target_critic(next_action)

            # Handle Q-values from multiple different target critic networks (if necessary)
            # (For double-Q, clip-Q, etc.)
            # next_qs = self.q_backup_strategy(next_qs)

            # assert next_qs.shape == (
            #     self.num_critic_networks,
            #     batch_size,
            # ), next_qs.shape

            if self.use_entropy_bonus and self.backup_entropy:
                # TODO(student): Add entropy bonus to the target values for SAC
                # next_action_entropy = ...
                # next_qs += ...
                next_action_entropy = -torch.mean(next_action_log_probs, dim=-1)
                next_qs += self.temperature * next_action_entropy

            # Compute the target Q-value
            # target_values: torch.Tensor = ...
            target_values = reward + self.discount * (1 - done.to(torch.float32)) * next_qs

            assert target_values.shape == (
                self.num_critic_networks,
                batch_size
            )

        # TODO(student): Update the critic
        # Predict Q-values
        # q_values = ...
        q_values = self.critic(next_action)
        assert q_values.shape == (self.num_critic_networks, batch_size), q_values.shape

        # Compute loss
        # loss: torch.Tensor = ...
        loss = torch.stack([self.critic_loss(q_value, target_values) for q_value in q_values]).sum()

        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

        return {
            "critic_loss": loss.item(),
            "q_values": q_values.mean().item(),
            "target_values": target_values.mean().item(),
        }

    def entropy(self, action_distribution: torch.distributions.Distribution):
        """
        Compute the (approximate) entropy of the action distribution for each batch element.
        """

        # TODO(student): Compute the entropy of the action distribution.
        # Note: Think about whether to use .rsample() or .sample() here...
        # return ...
        return -torch.mean(action_distribution.log_prob(action_distribution.rsample()), dim=-1)

    def get_actions(self, obs: torch.Tensor, num_samples=1):
        batch_size = obs.shape[0]
        temperature = 0.7
        log_probs = []
        state_actions = []
        for i in range(batch_size):
            input_ids = torch.stack(num_samples*[obs[i][obs[i]!=self.tokenizer.eos_token_id]]).squeeze()
            log_probs1 = torch.zeros(num_samples).cuda()
            for _ in range(self.max_length-input_ids.shape[1]):
                # Generate the next token
                logits = self.actor(input_ids).logits[:, -1, :] / temperature
                softmax_probs = torch.softmax(logits, dim=-1)
                distribution = torch.distributions.Categorical(probs=softmax_probs)
                next_token = distribution.sample()
                log_probs1 += torch.log(softmax_probs.gather(1,next_token.unsqueeze(0)).squeeze())
                input_ids = torch.cat([input_ids, next_token.unsqueeze(1)], dim=-1)
            log_probs.append(log_probs1)
            state_actions.append(input_ids)
        log_probs = torch.stack(log_probs)
        state_actions = torch.cat(state_actions)
        return state_actions, log_probs
    
    def actor_loss_reinforce(self, obs: torch.Tensor):
        batch_size = obs.shape[0]
        state_actions, log_probs = self.get_actions(obs, num_samples=self.num_actor_samples)
        with torch.no_grad():
            q_values = self.critic(state_actions)
            assert q_values.shape == (
                self.num_critic_networks,
                self.num_actor_samples*batch_size,
            ), q_values.shape

            # Our best guess of the Q-values is the mean of the ensemble
            q_values = torch.mean(q_values, axis=0)
            advantage = q_values

        # Do REINFORCE: calculate log-probs and use the Q-values
        # TODO(student)
        # log_probs = ...
        # loss = ...
        loss = -torch.mean(log_probs * advantage)

        return loss, -torch.mean(log_probs, dim=-1)

    def update_actor(self, obs: torch.Tensor):
        """
        Update the actor by one gradient step using either REPARAMETRIZE or REINFORCE.
        """
        loss, entropy = self.actor_loss_reinforce(obs)

        # Add entropy if necessary
        if self.use_entropy_bonus:
            loss -= self.temperature * entropy

        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        return {"actor_loss": loss.item(), "entropy": entropy.item()}

    def update_target_critic(self):
        self.soft_update_target_critic(1.0)

    def soft_update_target_critic(self, tau):
        for target_critic, critic in zip(self.target_critics, self.critics):
            for target_param, param in zip(
                target_critic.parameters(), critic.parameters()
            ):
                target_param.data.copy_(
                    target_param.data * (1.0 - tau) + param.data * tau
                )

    def update(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
        step: int,
    ):
        """
        Update the actor and critic networks.
        """

        # TODO(student): Update the critic for num_critic_upates steps, and add the output stats to critic_infos
        # critic_infos = []
        critic_infos = [self.update_critic(observations, actions, rewards, next_observations, dones) for _ in range(self.num_critic_updates)]

        # TODO(student): Update the actor
        # actor_info = ...
        actor_info = self.update_actor(observations)

        # TODO(student): Perform either hard or soft target updates.
        # Relevant variables:
        #  - step
        #  - self.target_update_period (None when using soft updates)
        #  - self.soft_target_update_rate (None when using hard updates)
        if self.target_update_period and step % self.target_update_period == 0:
            self.update_target_critic()
        elif self.soft_target_update_rate:
            self.soft_update_target_critic(self.soft_target_update_rate)

        # Average the critic info over all of the steps
        critic_info = {
            k: np.mean([info[k] for info in critic_infos]) for k in critic_infos[0]
        }

        # Deal with LR scheduling
        self.actor_lr_scheduler.step()
        self.critic_lr_scheduler.step()

        return {
            **actor_info,
            **critic_info,
            "actor_lr": self.actor_lr_scheduler.get_last_lr()[0],
            "critic_lr": self.critic_lr_scheduler.get_last_lr()[0],
        }

