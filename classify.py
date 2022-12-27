import os
import shutil
import numpy as np
import pandas as pd

path_img = 'E:\MaxImportant\learning\code&exe\DeepL\\trainimages'

men_path = 'E:\MaxImportant\learning\code&exe\DeepL\data\\test\\1'
women_path = 'E:\MaxImportant\learning\code&exe\DeepL\data\\test\\0'

ls = os.listdir(path_img)
lenl = len(ls)
print(len(ls))

df = pd.read_csv('E:\MaxImportant\learning\code&exe\DeepL\\train.csv')


for i in ls:
    # if i.find('testnan')!=-1:
    if df['is_male'][int(i[0:4])] == 1:
        shutil.move(path_img + '\\' + str(i), men_path+ '\\' + str(i))
    elif df['is_male'][int(i[0:4])] == -1:
        shutil.move(path_img + '\\' + str(i), women_path+ '\\' + str(i))
    # if int(i[0:4]) >= 2698:
    #     break



