import re
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

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
    # logdir is the path towards the stderr.log during training phase
    # parser.add_argument('logdir', type=str)
    # args = parser.parse_args()

    # score_dict = get_score_data(args.logdir)
    score_dict = get_score_data('stderr.log')
    
    plt.style.use('seaborn')
    mpl.rcParams['figure.figsize'] = (10, 10)

    n_rows = 2

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True)
    axs = [ax1, ax2, ax3, ax4]
    for idx, (ax, (title, rewards)) in enumerate(zip(axs, score_dict.items())):
        samples = np.arange(len(rewards)) * 14400 // 8
        ax.plot(samples, rewards)
        ax.set_title(title, fontsize=18)
    
    plt.ylabel("Reward Received", fontsize = 18)
    plt.xlabel("Number of Samples", fontsize = 18)
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    # for idx, (title, list) in enumerate(score_dict.items()):
    #         plt.subplot(n_rows, 2, 1 + idx)
    #         samples = np.arange(len(list)) * 14400 // 8
    #         plt.ticklabel_format(axis="x", style="plain", scilimits=(0,0))

    #         plt.plot(samples, list)
    #         plt.title(title, fontsize = 18)
    #         plt.xticks(size = 18)
    #         plt.yticks(size = 18)
    #         if idx == 0:
    #             plt.ylabel("Reward Received", fontsize = 18)
    #         elif idx == 2:
    #             plt.ylabel("Reward Received", fontsize = 18)
    #             plt.xlabel("Number of Samples", fontsize = 18)
    #         elif idx == 3:
    #             plt.xlabel("Number of Samples", fontsize = 18)
           

    plt.tight_layout()
    plt.show()
    # filename = args.logdir.split('/')[-1]

    # parent_path = os.path.abspath(os.path.join(args.logdir, os.pardir))
    if os.path.isdir(parent_path + "/pictures/") == False:
        os.mkdir(parent_path + "/pictures/")
        plt.savefig(fname = "{parent_path}/pictures/{fname}.png".format(parent_path = parent_path ,fname = "trainning result"))
    else:
        plt.savefig(fname = "{parent_path}/pictures/{fname}.png".format(parent_path = parent_path ,fname = "trainning result"))


    
    

