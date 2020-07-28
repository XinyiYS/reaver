import re
import argparse
import numpy as np

def get_score_data(log_file_path):
    regex = 'score: \[.*\]'

    # get expressions like "score: [0]" and store them in this list
    with open(log_file_path, "r") as file:
        score_list = []
        for line in file:
            for match in re.finditer(regex, line, re.S):
                match_text = match.group()
                score = [int(s) for s in re.findall(r'\b\d+\b', match_text)][0]
                score_list.append(score)
                
    file.closed
    return score_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('logdir', type=str)
    args = parser.parse_args()

    score_list = get_score_data(args.logdir)
    #print("score list is ", score_list)
    print("Total game step is ", len(score_list) * 14400)
    print("Total samples is", len(score_list) * 14400 / 8)

