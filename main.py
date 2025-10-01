import gymnasium as gym
import highway_env
import time
from agent import DQNAgent


def train():
    env = gym.make("highway-v0", render_mode=None)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim)

    episodes = 200
    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0

        for _ in range(200):
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.remember(state, action, reward, next_state, done)
            agent.train_step()

            state = next_state
            total_reward += reward

            if done:
                break

        agent.update_target()
        print(f"Episode {ep}, reward: {total_reward:.2f}, epsilon: {agent.epsilon:.2f}")

    env.close()
    return agent


def demo(agent):
    env = gym.make("highway-v0", render_mode="human")
    state, _ = env.reset()
    done = False
    total_reward = 0

    print("\n🚗 Running demo episode...")
    while not done:
        action = agent.act(state, use_random=False)  # greedy policy
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward

        env.render()
        time.sleep(0.05)

        state = next_state

    print(f"✅ Demo finished - Total Reward: {total_reward:.2f}")
    env.close()


if __name__ == "__main__":
    agent = train()
    demo(agent)
