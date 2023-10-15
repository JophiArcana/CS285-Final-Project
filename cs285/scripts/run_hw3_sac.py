import argparse
import pickle
import time

import gym
import numpy as np
import torch
import tqdm
from matplotlib import pyplot as plt

from cs285.agents.soft_actor_critic import SoftActorCritic
from cs285.infrastructure import pytorch_util as ptu
from cs285.infrastructure import utils
from cs285.infrastructure.logger import Logger
from cs285.infrastructure.replay_buffer import ReplayBuffer
from scripting_utils import make_logger, make_config


def run_training_loop(config: dict, logger: Logger, args: argparse.Namespace):
    # set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ptu.init_gpu(use_gpu=not args.no_gpu, gpu_id=args.which_gpu)

    # make the gym environment
    env = config["make_env"]()
    eval_env = config["make_env"]()
    render_env = config["make_env"](render=True)

    ep_len = config["ep_len"] or env.spec.max_episode_steps

    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    assert (
        not discrete
    ), "Our actor-critic implementation only supports continuous action spaces. (This isn't a fundamental limitation, just a current implementation decision.)"

    ob_shape = env.observation_space.shape
    ac_dim = env.action_space.shape[0]

    # simulation timestep, will be used for video saving
    if "model" in dir(env):
        fps = 1 / env.model.opt.timestep
    else:
        fps = env.env.metadata["render_fps"]

    # initialize agent
    agent = SoftActorCritic(
        ob_shape,
        ac_dim,
        **config["agent_kwargs"],
    )

    replay_buffer = ReplayBuffer(config["replay_buffer_capacity"])

    observation = env.reset()

    return_logs, start_time = [], time.time()
    exp_name = args.exp_name
    if exp_name is None and "exp_name" in config:
        exp_name = config["exp_name"]
    for step in tqdm.trange(config["total_steps"], dynamic_ncols=True):
        if step < config["random_steps"]:
            action = env.action_space.sample()
        else:
            # DONE(student): Select an action
            action = agent.get_action(observation)

        # Step the environment and add the data to the replay buffer
        next_observation, reward, done, info = env.step(action)
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

        # Train the agent
        batch, update_info = None, None
        if step >= config["training_starts"]:
            # DONE(student): Sample a batch of config["batch_size"] transitions from the replay buffer
            batch = replay_buffer.sample(config["batch_size"])
            update_info = update_info = agent.update(
                ptu.from_numpy(batch["observation"]),
                ptu.from_numpy(batch["action"]),
                ptu.from_numpy(batch["reward"]),
                ptu.from_numpy(batch["next_observation"]),
                ptu.from_numpy(batch["done"]),
                step
            )

            # Logging
            update_info["actor_lr"] = agent.actor_lr_scheduler.get_last_lr()[0]
            update_info["critic_lr"] = agent.critic_lr_scheduler.get_last_lr()[0]

            if step % args.log_interval == 0:
                for k, v in update_info.items():
                    logger.log_scalar(v, k, step)
                logger.flush()

        # Run evaluation
        if step % args.eval_interval == 0:
            eval_trajs = utils.sample_n_trajectories(
                eval_env,
                policy=agent,
                ntraj=args.num_eval_trajectories,
                max_length=ep_len,
            )
            returns = [t["episode_statistics"]["r"] for t in eval_trajs]
            ep_lens = [t["episode_statistics"]["l"] for t in eval_trajs]

            if step >= config["training_starts"]:
                train_trajs = [{k: v[t] for k, v in batch.items()} for t in range(config["batch_size"])]
                logs = update_info
                logs.update(utils.compute_metrics(train_trajs, eval_trajs))
                logs["Train_EnvstepsSoFar"] = step
                logs["TimeSinceStart"] = time.time() - start_time
                if step == 0:
                    logs["Initial_DataCollection_AverageReturn"] = logs["Train_AverageReturn"]
                return_logs.append(logs)

            if exp_name is not None and len(return_logs) > 0 and step % args.save_frequency == 0:
                current_return_logs = {k: np.array([d.get(k, return_logs[0][k]) for d in return_logs]) for k in return_logs[0]}
                with open(f'plot_data/{exp_name}.pkl', 'wb') as fp:
                    pickle.dump(current_return_logs, fp)

            logger.log_scalar(np.mean(returns), "eval_return", step)
            logger.log_scalar(np.mean(ep_lens), "eval_ep_len", step)

            if len(returns) > 1:
                logger.log_scalar(np.std(returns), "eval/return_std", step)
                logger.log_scalar(np.max(returns), "eval/return_max", step)
                logger.log_scalar(np.min(returns), "eval/return_min", step)
                logger.log_scalar(np.std(ep_lens), "eval/ep_len_std", step)
                logger.log_scalar(np.max(ep_lens), "eval/ep_len_max", step)
                logger.log_scalar(np.min(ep_lens), "eval/ep_len_min", step)

            if args.num_render_trajectories > 0:
                video_trajectories = utils.sample_n_trajectories(
                    render_env,
                    agent,
                    args.num_render_trajectories,
                    ep_len,
                    render=True,
                )

                logger.log_paths_as_videos(
                    video_trajectories,
                    step,
                    fps=fps,
                    max_videos_to_save=args.num_render_trajectories,
                    video_title="eval_rollouts",
                )
    return return_logs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", "-cfg", type=str, required=True)
    parser.add_argument("--exp_name", type=str, default=None)

    parser.add_argument("--eval_interval", "-ei", type=int, default=5000)
    parser.add_argument("--num_eval_trajectories", "-neval", type=int, default=10)
    parser.add_argument("--num_render_trajectories", "-nvid", type=int, default=0)

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no_gpu", "-ngpu", action="store_true")
    parser.add_argument("--which_gpu", "-g", default=0)
    parser.add_argument("--video_log_freq", type=int, default=-1)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--save_frequency", type=int, default=10000)

    args = parser.parse_args()

    # create directory for logging
    logdir_prefix = "hw3_sac_"  # keep for autograder

    config = make_config(args.config_file)
    logger = make_logger(logdir_prefix, config)

    return_logs = run_training_loop(config, logger, args)
    ####################################################################################################################
    return_logs = {k: np.array([d.get(k, return_logs[0][k]) for d in return_logs]) for k in return_logs[0]}

    plt.plot(return_logs['Train_EnvstepsSoFar'], return_logs['Eval_AverageReturn'], label='Eval_AverageReturn')
    plt.show()

    exp_name = args.exp_name
    if exp_name is None and "exp_name" in config:
        exp_name = config["exp_name"]
    if exp_name is not None:
        with open(f'plot_data/{exp_name}.pkl', 'wb') as fp:
            pickle.dump(return_logs, fp)


if __name__ == "__main__":
    main()
