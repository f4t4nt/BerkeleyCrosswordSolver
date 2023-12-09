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

# from cryptic_dataset import CrypticDataset

import argparse
import re

from crossword_env import CrosswordEnv
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from load import load_data

def run_training_loop(config: dict, logger: Logger, args: argparse.Namespace, env_data, eval_data):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ptu.init_gpu(use_gpu=not args.no_gpu, gpu_id=args.which_gpu)

    # USELESS ###########
    # env = config["make_env"]()
    # eval_env = config["make_env"]()
    # render_env = config["make_env"](render=True)
    #####################

    # ep_len = config["ep_len"] or env.spec.max_episode_steps
    ep_len = 1000 # TODO: change as needed
    batch_size = config["batch_size"] or batch_size

    # USELESS ###########
    # # simulation timestep, will be used for video saving
    # if "model" in dir(env):
    #     fps = 1 / env.model.opt.timestep
    # else:
    #     fps = env.env.metadata["render_fps"]
    #####################

    # initialize agent
    # agent = SoftActorCritic(
    #     **config["agent_kwargs"],
    # )

    # replay_buffer = ReplayBuffer(config["replay_buffer_capacity"])

    env = CrosswordEnv(env_data)
    eval_env = CrosswordEnv(eval_data)

    observation = env.reset()
    
    test_actions = env.obs_str
    # append random words to the end of each string
    import random
    import string
    for i, s in enumerate(test_actions):
        test_actions[i] = s + " " + " ".join([random.choice(string.ascii_lowercase) for _ in range(10)])
    tokenized = env.tokenizer(test_actions, padding=True, truncation=True, return_tensors="pt").input_ids.cuda()
    next_observation, reward, done, info = env.step(tokenized)
    print(next_observation)

    for step in tqdm.trange(config["total_steps"], dynamic_ncols=True):
        action = agent.get_action(observation)

        next_observation, reward, done, info = env.step(action) # TODO: calculate reward
        replay_buffer.insert(
            observation=observation,
            action=action,
            reward=reward,
            next_observation=next_observation,
            # done=done and not info.get("TimeLimit.truncated", False),
            done=done,
        )

        if done:
            # logger.log_scalar(info["episode"]["r"], "train_return", step)
            # logger.log_scalar(info["episode"]["l"], "train_ep_len", step)
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

        # if step % args.eval_interval == 0:
        #     trajectories = utils.sample_n_trajectories(
        #         eval_env, #TODO: do stuff from eval environment
        #         policy=agent,
        #         ntraj=args.num_eval_trajectories,
        #         max_length=ep_len,
        #     )
        #     returns = [t["episode_statistics"]["r"] for t in trajectories]
        #     ep_lens = [t["episode_statistics"]["l"] for t in trajectories]

        #     logger.log_scalar(np.mean(returns), "eval_return", step)
        #     logger.log_scalar(np.mean(ep_lens), "eval_ep_len", step)

        #     if len(returns) > 1:
        #         logger.log_scalar(np.std(returns), "eval/return_std", step)
        #         logger.log_scalar(np.max(returns), "eval/return_max", step)
        #         logger.log_scalar(np.min(returns), "eval/return_min", step)
        #         logger.log_scalar(np.std(ep_lens), "eval/ep_len_std", step)
        #         logger.log_scalar(np.max(ep_lens), "eval/ep_len_max", step)
        #         logger.log_scalar(np.min(ep_lens), "eval/ep_len_min", step)


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

    # with open("bruteforce.json", "r") as f:
    #     data = json.load(f)
    # data = data["data"]
    # train_data = CrypticDataset(data[:100])
    # test_data = CrypticDataset(data[100:])
    
    def clean_string(s):
        return re.sub(r'[^a-zA-Z0-9\s]', '', s).lower()
    
    def has_digit(s):
        return any(c.isdigit() for c in s)
    
    data = load_data(randomize=True)
    # data = [(d[1], d[2], d[3], int(d[4])) for d in data if
    data = [(clean_string(d[1]), clean_string(d[2]), clean_string(d[3]), int(d[4])) for d in data if
            len(d[1]) > 0 and
            len(d[2]) > 0 and
            len(d[3]) > 0 and
            len(d[4]) > 0 and
            not has_digit(d[1]) and
            not has_digit(d[2]) and
            not has_digit(d[3]) and
            d[3].count(' ') == 0 and
            d[4].isdigit()
        ]
    env_data = data[:int(len(data) * 0.8)]
    eval_data = data[int(len(data) * 0.8):]

    run_training_loop(config, logger, args, env_data, eval_data)

if __name__ == "__main__":
    main()
