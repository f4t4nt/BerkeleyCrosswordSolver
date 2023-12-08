import os

import os
import sys

# from cs285.agents.soft_actor_critic import SoftActorCritic
# from cs285.infrastructure.replay_buffer import ReplayBuffer
from soft_actor_critic import SoftActorCritic
from replay_buffer import ReplayBuffer

import os
import time

import gym
import numpy as np
import torch
# from cs285.infrastructure import pytorch_util as ptu
import pytorch_util as ptu
import tqdm

# from cs285.infrastructure import utils
# from cs285.infrastructure.logger import Logger
import utils
from logger import Logger

from scripting_utils import make_logger, make_config

from cryptic_dataset import CrypticDataset

import argparse
import json

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
import models

dpr = models.setup_closedbook(0)
def reward_function(currents, targets):
    score = dpr.get_scores(currents, targets)
    for i, s in enumerate(currents):
        score += (len(s) - len(targets[i])) ** 2
    for i, s in enumerate(currents):
        alpha_c = np.zeros(26)
        for c in s:
            alpha_c[ord(c) - ord('a')] += 1
        for c in targets[i]:
            alpha_c[ord(c) - ord('a')] -= 1
        score += np.sum(alpha_c ** 2)
    return score

def run_training_loop(config: dict, logger: Logger, args: argparse.Namespace, train_data: CrypticDataset, test_data: CrypticDataset):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ptu.init_gpu(use_gpu=not args.no_gpu, gpu_id=args.which_gpu)

    # USELESS ###########
    # env = config["make_env"]()
    # eval_env = config["make_env"]()
    # render_env = config["make_env"](render=True)
    #####################

    ep_len = config["ep_len"] or env.spec.max_episode_steps
    batch_size = config["batch_size"] or batch_size

    # USELESS ###########
    # # simulation timestep, will be used for video saving
    # if "model" in dir(env):
    #     fps = 1 / env.model.opt.timestep
    # else:
    #     fps = env.env.metadata["render_fps"]
    #####################

    # initialize agent
    agent = SoftActorCritic(
        **config["agent_kwargs"],
    )

    replay_buffer = ReplayBuffer(config["replay_buffer_capacity"])

    observation = env.reset()

    for step in tqdm.trange(config["total_steps"], dynamic_ncols=True):
        action = agent.get_action(observation)

        next_observation, reward, done, info = env.step(action) # TODO: calculate reward
        replay_buffer.insert(
            observation=observation,
            action=action,
            reward=reward,
            next_observation=next_observation,
            done=done and not info.get("TimeLimit.truncated", False),
        )

        if done:
            logger.log_scalar(info["episode"]["r"], "train_return", step)
            logger.log_scalar(info["episode"]["l"], "train_ep_len", step)
            observation = env.reset()
        else:
            observation = next_observation

        if step >= config["training_starts"]:
            batch = ptu.from_numpy(replay_buffer.sample(config["batch_size"]))
            update_info = agent.update(
                observations=batch["observations"],
                actions=batch["actions"],
                rewards=batch["rewards"],
                next_observations=batch["next_observations"],
                dones=batch["dones"],
                step=step
            )

            update_info["actor_lr"] = agent.actor_lr_scheduler.get_last_lr()[0]
            update_info["critic_lr"] = agent.critic_lr_scheduler.get_last_lr()[0]

            if step % args.log_interval == 0:
                for k, v in update_info.items():
                    logger.log_scalar(v, k, step)
                    logger.log_scalars
                logger.flush()

        if step % args.eval_interval == 0:
            trajectories = utils.sample_n_trajectories(
                eval_env, #TODO: do stuff from eval environment
                policy=agent,
                ntraj=args.num_eval_trajectories,
                max_length=ep_len,
            )
            returns = [t["episode_statistics"]["r"] for t in trajectories]
            ep_lens = [t["episode_statistics"]["l"] for t in trajectories]

            logger.log_scalar(np.mean(returns), "eval_return", step)
            logger.log_scalar(np.mean(ep_lens), "eval_ep_len", step)

            if len(returns) > 1:
                logger.log_scalar(np.std(returns), "eval/return_std", step)
                logger.log_scalar(np.max(returns), "eval/return_max", step)
                logger.log_scalar(np.min(returns), "eval/return_min", step)
                logger.log_scalar(np.std(ep_lens), "eval/ep_len_std", step)
                logger.log_scalar(np.max(ep_lens), "eval/ep_len_max", step)
                logger.log_scalar(np.min(ep_lens), "eval/ep_len_min", step)

            # USELESS ###########
            # if args.num_render_trajectories > 0:
            #     video_trajectories = utils.sample_n_trajectories(
            #         render_env,
            #         agent,    
            #         args.num_render_trajectories,
            #         ep_len,   
            #         render=True,
            #     )             
                                
            #     logger.log_paths_as_videos(
            #         video_trajectories,
            #         step,     
            #         fps=fps,  
            #         max_videos_to_save=args.num_render_trajectories,
            #         video_title="eval_rollouts",
            #     )             
            ####################


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--config_file", "-cfg", type=str, required=True)
    parser.add_argument("--config_file", "-cfg", type=str, default="285_final/hw3/sanity_invertedpendulum_reinforce.yaml")

    parser.add_argument("--eval_interval", "-ei", type=int, default=5000)
    parser.add_argument("--num_eval_trajectories", "-neval", type=int, default=10)
    parser.add_argument("--num_render_trajectories", "-nvid", type=int, default=0)

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no_gpu", "-ngpu", action="store_true")
    parser.add_argument("--which_gpu", "-g", default=0)
    parser.add_argument("--log_interval", type=int, default=1000)

    args = parser.parse_args()

    logdir_prefix = "hw3_sac_"

    config = make_config(args.config_file)
    logger = make_logger(logdir_prefix, config)

    with open("bruteforce.json", "r") as f:
        data = json.load(f)
    data = data["data"]
    train_data = CrypticDataset(data[:100])
    test_data = CrypticDataset(data[100:])

    run_training_loop(config, logger, args, train_data, test_data)


if __name__ == "__main__":
    main()
