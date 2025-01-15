from gymnasium.envs.registration import make, pprint_registry, register, registry, spec
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym

def show_state(env, episode=0, step=0, info=""):
    pass
    # plt.figure(3)
    # plt.clf()
    # plt.imshow(env.render())#mode='rgb_array'))
    # plt.title("%s | Eposide: %d | Step: %d %s" % ('Cart-pole-v1', episode, step, info))
    # plt.axis('off')

    # # display.clear_output(wait=True)
    # # display.display(plt.gcf())
    # plt.show()


device = "cpu"
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Execution parameters
SHOW_ANIMATION = False
EPISODES_MAX = 300

# TIME_MAX = 10
# STEPS_MAX = int(TIME_MAX/env.unwrapped.model.opt.timestep)
STEPS_MAX = 3000

DESIRED_RETURN = 3000

register(
    id="Hexapod-v0",
    entry_point="gym_env:HexapodEnv",
    max_episode_steps=STEPS_MAX,
    reward_threshold=3800.0,
)
env = gym.make('Hexapod-v0', render_mode='human',max_geom=7000)

print('SHApes',env.observation_space.shape, env.action_space.shape)



# Loggers
log_steps_number = np.zeros(EPISODES_MAX)

# PQ
for i_episode in range(EPISODES_MAX):
    observation = env.reset()[0]
    state = observation

    # show results
    if (i_episode + 1) % 20 == 0:
        # plt.figure(1)
        # plt.clf()
        # plt.plot([0,i_episode], [495, 495], label="threshold")
        # plt.plot(range(0,i_episode), log_steps_number[0:i_episode], label="solution 1")
        # plt.xlabel('episode')
        # plt.ylabel('episode steps')
        # plt.legend()
        # # display.clear_output(wait=True)
        # plt.show()
        pass

    for t in range(STEPS_MAX):
        # print(t)
        action, _, _ = np.zeros(8),0,0#agent.get_action(state)
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        state = observation

        if done:
            log_steps_number[i_episode] = t;
            break

print("done")