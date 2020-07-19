import re
import argparse
import numpy as np

def get_reward_data(log_file_path):
    regex = 'reward: \[.\]'

    # get expressions like "reward: [0]" and store them in this list
    match_list = []
    with open(log_file_path, "r") as file:
        for line in file:
            for match in re.finditer(regex, line, re.S):
                match_text = match.group()
                match_list.append(match_text)

        reward_list = []
        for expressions in match_list:
            # get the number
            reward = int(expressions[9])
            reward_list.append(reward)

        return reward_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('logdir', type=str)
    args = parser.parse_args()

    reward_list = get_reward_data(args.logdir)
    print("Mean is ", np.mean(reward_list))    
    print("Std deviation is ", np.std(reward_list))
