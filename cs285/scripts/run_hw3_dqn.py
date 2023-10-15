import argparse
import pickle
import time

import gym
import numpy as np
import torch
import tqdm
from matplotlib import pyplot as plt

from cs285.agents.dqn_agent import DQNAgent
from cs285.infrastructure import pytorch_util as ptu
from cs285.infrastructure import utils
from cs285.infrastructure.logger import Logger
from cs285.infrastructure.replay_buffer import MemoryEfficientReplayBuffer, ReplayBuffer
from scripting_utils import make_logger, make_config

MAX_NVIDEO = 2


def run_training_loop(config: dict, logger: Logger, args: argparse.Namespace):
    # set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ptu.init_gpu(use_gpu=not args.no_gpu, gpu_id=args.which_gpu)

    print(f'make_critic: {config["agent_kwargs"]["make_critic"]}')

    # make the gym environment
    env = config["make_env"]()
    eval_env = config["make_env"]()
    render_env = config["make_env"](render=True)
    exploration_schedule = config["exploration_schedule"]
    discrete = isinstance(env.action_space, gym.spaces.Discrete)

    assert discrete, "DQN only supports discrete action spaces"

    agent = DQNAgent(
        env.observation_space.shape,
        env.action_space.n,
        **config["agent_kwargs"],
    )

    # simulation timestep, will be used for video saving
    if "model" in dir(env):
        fps = 1 / env.model.opt.timestep
    elif "render_fps" in env.env.metadata:
        fps = env.env.metadata["render_fps"]
    else:
        fps = 4

    ep_len = env.spec.max_episode_steps

    observation = None

    # Replay buffer
    if len(env.observation_space.shape) == 3:
        stacked_frames = True
        frame_history_len = env.observation_space.shape[0]
        assert frame_history_len == 4, "only support 4 stacked frames"
        replay_buffer = MemoryEfficientReplayBuffer(
            frame_history_len=frame_history_len
        )
    elif len(env.observation_space.shape) == 1:
        stacked_frames = False
        replay_buffer = ReplayBuffer()
    else:
        raise ValueError(
            f"Unsupported observation space shape: {env.observation_space.shape}"
        )

    def reset_env_training():
        nonlocal observation

        observation = env.reset()

        assert not isinstance(
            observation, tuple
        ), "env.reset() must return np.ndarray - make sure your Gym version uses the old step API"
        observation = np.asarray(observation)

        if isinstance(replay_buffer, MemoryEfficientReplayBuffer):
            replay_buffer.on_reset(observation=observation[-1, ...])

    reset_env_training()

    return_logs, start_time = [], time.time()
    exp_name = args.exp_name
    if exp_name is None and "exp_name" in config:
        exp_name = config["exp_name"]
    for step in tqdm.trange(config["total_steps"], dynamic_ncols=True):
        epsilon = exploration_schedule.value(step)

        # DONE(student): Compute action
        action = agent.get_action(observation, epsilon)

        # DONE(student): Step the environment
        next_observation, reward, done, info = env.step(action)
        next_observation = np.asarray(next_observation)
        truncated = info.get("TimeLimit.truncated", False)

        # DONE(student): Add the data to the replay buffer
        if isinstance(replay_buffer, MemoryEfficientReplayBuffer):
            # We're using the memory-efficient replay buffer,
            # so we only insert next_observation (not observation)
            replay_buffer.insert(np.array([action]), reward, next_observation[-1], ~truncated and done)
        else:
            # We're using the regular replay buffer
            replay_buffer.insert(observation, np.array([action]), reward, next_observation[-1], ~truncated and done)

        # Handle episode termination
        if truncated or done:
            reset_env_training()

            logger.log_scalar(info["episode"]["r"], "train_return", step)
            logger.log_scalar(info["episode"]["l"], "train_ep_len", step)
        else:
            observation = next_observation

        # Main DQN training loop
        batch, update_info = None, None
        if step >= config["training_starts"]:
            # DONE(student): Sample config["batch_size"] samples from the replay buffer
            batch = replay_buffer.sample(config["batch_size"])

            # Convert to PyTorch tensors
            batch = ptu.from_numpy(batch)

            # DONE(student): Train the agent. `batch` is a dictionary of numpy arrays
            update_info = agent.update(
                batch["observation"],
                batch["action"],
                batch["reward"],
                batch["next_observation"],
                batch["done"],
                step
            )

            # Logging code
            update_info["epsilon"] = epsilon
            update_info["lr"] = agent.lr_scheduler.get_last_lr()[0]

            if step % args.log_interval == 0:
                for k, v in update_info.items():
                    logger.log_scalar(v, k, step)
                logger.flush()

        if step % args.eval_interval == 0:
            # Evaluate
            eval_trajs = utils.sample_n_trajectories(
                eval_env,
                agent,
                args.num_eval_trajectories,
                ep_len,
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

    parser.add_argument("--eval_interval", "-ei", type=int, default=10000)
    parser.add_argument("--num_eval_trajectories", "-neval", type=int, default=10)
    parser.add_argument("--num_render_trajectories", "-nvid", type=int, default=0)

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no_gpu", "-ngpu", action="store_true")
    parser.add_argument("--which_gpu", "-gpu_id", default=0)
    parser.add_argument("--video_log_freq", type=int, default=-1)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--save_frequency", type=int, default=10000)

    args = parser.parse_args()

    # create directory for logging
    logdir_prefix = "hw3_dqn_"  # keep for autograder

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
