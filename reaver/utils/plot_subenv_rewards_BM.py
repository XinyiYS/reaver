import re
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


SUBENV_LIST = ['BuildSupplyDepots', 'BuildBarracks', 'BuildMarinesWithBarracks','BuildMarines']
def get_score_data(log_file_path):
    
    score_regex = 'score: \[.*\]'

    score_dict = {
        "BuildSupplyDepots": [],
        "BuildBarracks": [],
        "BuildMarinesWithBarracks": [],
        "BuildMarines": []
    }

    subenv_scope = 0
    with open(log_file_path, "r") as file:
        for line in file:
            # check which is the current sub_env 
            subenv_scope_found = False
            for i in range(4):
                for match in re.finditer(SUBENV_LIST[i], line, re.S):
                    if len(match.group(0)) != 0:
                        subenv_scope = i
                        subenv_scope_found = True
                        break
                # if found the correct scope then break
                if subenv_scope_found:
                    break

            # get the score and store in seperate lists
            for match in re.finditer(score_regex, line, re.S):
                match_text = match.group()
                # print("match text is " , match_text)
                score = [int(s) for s in re.findall(r'\b\d+\b', match_text)][0]
                # print("score is ", score)
                score_dict[SUBENV_LIST[subenv_scope]].append(score)

    file.closed    
    #print(score_dict)
    return score_dict

            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('logdir', type=str)
    args = parser.parse_args()

    score_dict = get_score_data(args.logdir)
    
    plt.style.use('seaborn')
    mpl.rcParams['figure.figsize'] = (10, 10)

    n_rows = 4

    plt.style.use('seaborn')

    for idx, (title, list) in enumerate(score_dict.items()):
        plt.subplot(n_rows,1, 1 + idx)
        plt.plot(list)
        plt.title(title)

    plt.tight_layout()
    plt.show()
    plt.savefig(fname = "./HRL_result_plot.png")


    
    

