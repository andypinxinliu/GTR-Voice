import os
import pdb
import torch
import sys



# stats pitch mean std of all pitch file in a folder
# input: folder path
# output: mean and std of pitch for all pitch files in the folder
def stats_pitch_meanstd(folder_path):
    # get all pitch files in the folder
    pitch_files = [f for f in os.listdir(folder_path) if f.endswith('.pt')]
    # get all pitch values from pitch files
    pitch_values = []
    for pitch_file in pitch_files:
        # load torch tensor of pitch values from pitch file .pt
        pitch = torch.load(os.path.join(folder_path, pitch_file))
        # convert to list
        pitch = pitch.tolist()[0]
        # get pitch values without 0
        pitch_values += [x for x in pitch if x != 0.0]


    # calculate mean and std of pitch values
    print(len(pitch_values))
    mean = sum(pitch_values) / len(pitch_values)
    std = (sum([(x - mean) ** 2 for x in pitch_values]) / len(pitch_values)) ** 0.5
    return mean, std

# test
folder_path = sys.argv[1]
mean, std = stats_pitch_meanstd(folder_path)
print('Mean:', mean)
print('Std:', std)