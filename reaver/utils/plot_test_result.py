import re
import argparse
import numpy as np

def get_score_data(log_file_path):
    regex = 'score: \[.\]'

    # get expressions like "score: [0]" and store them in this list
    match_list = []
    with open(log_file_path, "r") as file:
        for line in file:
            for match in re.finditer(regex, line, re.S):
                match_text = match.group()
                match_list.append(match_text)

        score_list = []
        for expressions in match_list:
            # get the number
            score = int(expressions[8])
            score_list.append(score)

        return score_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('logdir', type=str)
    args = parser.parse_args()

    score_list = get_score_data(args.logdir)
    # print("score list is ", score_list)
    print("Mean is ", np.mean(score_list))    
    print("Std deviation is ", np.std(score_list))
