# Playing Atari with Deep Reinforcement Learning (PyTorch Reimplementation)
This repository contains a PyTorch implementation of the Deep Reinforcement Learning algorithm for playing Atari games. The code is based on the paper "Playing Atari with Deep Reinforcement Learning" by Mnih et al. (2013). The implementation is designed to be simple and easy to understand, while still being efficient and effective.

ADD GIF OF AGENT HERE

## Installation
To install the required packages, you can use the following command:

```bash
pip install -r requirements.txt
```
## Usage
To train the agent on an Atari game, you can use the following command:

Replace `<ENV_NAME>` with the name of the Atari environment you want to train on (e.g., `ALE/Pong-v5`) and `<NUM_FRAMES>` with the number of frames you want to train for. Remember to include the ALE prefix for Atari environments.

```bash
python src/train.py --env <ENV_NAME> --num_frames <NUM_FRAMES>
```

To evaluate the agent, you can use the following command:

Replace `<ENV_NAME>` with the name of the Atari environment you want to evaluate on (e.g., `ALE/Pong-v5`), `<NUM_EPISODES>` with the number of episodes you want to evaluate for, and `<RENDER_MODE>` with the render mode (e.g., `human` or `rgb_array`).

```bash
python src/test_agent.py --env <ENV_NAME> --num_episodes <NUM_EPISODES> -- render_mode <RENDER_MODE>
```

## Directory Structure

```plaintext
.
├── README.md
├── requirements.txt
├── src
│   ├── agent.py
│   ├── config.py
│   ├── graph_results.py
│   ├── model.py
│   ├── test_all_agents.py
│   ├── test_agent.py
│   └── train_agent.py
├── logs_{ENV_NAME}
│   ├── {AGENT_NAME}_model_final.pth
│   ├── results_{AGENT_NAME}.csv
│   ├── avg_q_value_vs_episode.png
│   ├── avg_reward_vs_episode.png
│   ├── reward_vs_episode.png
│   └── reward_log.csv
```

## Configuration
The configuration was adjusted so the agent can be trained on a single GPU or on Apple Silicon using Metal. The configuration file is located in `src/config.py`. You can modify the hyperparameters and other settings in this file to suit your needs.

## Logging
The training and evaluation results are logged in the `logs_{ENV_NAME}` directory. The logs include the following files:
- `{AGENT_NAME}_model_final.pth`: The final model weights of the trained agent.
- `results_{AGENT_NAME}.csv`: The results of the training and evaluation, including average rewards and Q-values.
- `avg_q_value_vs_episode.png`: A plot of the average Q-value vs. episode.
- `avg_reward_vs_episode.png`: A plot of the average reward vs. episode.
- `reward_vs_episode.png`: A plot of the reward vs. episode.
- `reward_log.csv`: A CSV file containing the reward log for each episode.

## Results
will add results graphs and table here