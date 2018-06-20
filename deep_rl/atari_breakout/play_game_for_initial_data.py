""" Script to record human play of the Atari-breakout game """

import gym
import argparse
import sys
import tty
import termios
import torch as th
import pickle

device = th.device("cuda" if th.cuda.is_available() else "cpu")


# Code for getting the keyboard inputs
class _Getch:
    def __call__(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(3)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch


def get_action():
    inkey = _Getch()

    while True:
        k = inkey()
        if k != '':
            break

    if k == '\x1b[A':
        return 0  # Noop
    elif k == '\x1b[B':
        return 1  # fire
    elif k == '\x1b[C':
        return 2  # right
    elif k == '\x1b[D':
        return 3  # left
    else:
        return 0  # Noop by default


def play_and_record_game(env, memory):
    from _3_RL_DQ_Learning_Atari_breakout import preprocess_frame, \
        expand_and_convert, connect_frames
    from itertools import count

    current_screen = expand_and_convert(preprocess_frame(env.reset()))
    last_screen = th.zeros_like(current_screen, device=device, dtype=th.float)

    state = connect_frames(last_screen, current_screen)

    rewards = []  # initialize to empty list
    for t in count():
        # Select and perform an action
        env.render()
        action = get_action()
        obs, reward, done, _ = env.step(action)
        rewards.append(reward)
        reward = th.tensor([reward], device=device)

        # Observe new state
        last_screen = current_screen
        current_screen = expand_and_convert(preprocess_frame(obs))
        if not done:
            next_state = connect_frames(last_screen, current_screen)
        else:
            next_state = None

        # Store the transition in memory
        action = th.tensor([[action]], device=device, dtype=th.long)
        memory.push(state, action, next_state, reward)
        print(state)
        print(action)
        print(reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        if done:
            break

    # return the number of steps spent in the game and the total reward collected
    return t, sum(rewards)


def save_pickle(obj, file_name):
    """
    save the given data obj as a pickle file
    :param obj: python data object
    :param file_name: path of the output file
    :return: None (writes file to disk)
    """
    with open(file_name, 'wb') as dumper:
        pickle.dump(obj, dumper, pickle.HIGHEST_PROTOCOL)


def parse_arguments():
    """
    command line arguments parser
    :return: args => parsed command line arguments
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("--num_games", action="store", type=int,
                        default=1,
                        help="number of manual games")

    parser.add_argument("--out_file", action="store", type=str,
                        default="Saved_Models/preplay_memory.mem",
                        help="output file to collect the played memory")

    args = parser.parse_args()

    return args


def main(args):
    """
    main function of the script
    :param args: parsed command line arguments
    :return: None
    """
    from _3_RL_DQ_Learning_Atari_breakout import ReplayMemory

    env = gym.make("Breakout-v0")
    memory = ReplayMemory(50000)

    for _ in range(args.num_games):
        steps, total_reward = play_and_record_game(env, memory)

        print("Total game steps: %d  Total reward: %d" % (steps, total_reward))

    # close the environment
    env.close()

    # save the memory to the file
    print("Saving the memory at: ", args.out_file)
    save_pickle(memory, args.out_file)


if __name__ == '__main__':
    main(parse_arguments())
