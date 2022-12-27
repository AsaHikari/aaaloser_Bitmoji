import os
import pandas as pd

label_path = 'E:\MaxImportant\learning\code&exe\DeepL\\train.csv'

def generate(dir):
    df = pd.read_csv(label_path)
    # print(df['image_id'][0][0:4])
    files = os.listdir(dir) #os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表。
    files.sort()  #对文件或文件夹进行排序
    print('****************')
    print('input :', dir)
    print('start...')
    listText = open('E:\MaxImportant\learning\code&exe\DeepL\\test.txt', 'a+')  #创建并打开一个txt文件，a+表示打开一个文件并追加内容
    for file in files[2700:3000]:  #遍历文件夹中的文件
        # print(file[0:4])
        for i in range(0,3000):
            if file[0:4] == df['image_id'][i][0:4]:
                label = df['is_male'][i]
        fileType = os.path.split(file) #os.path.split（）返回文件的路径和文件名，【0】为路径，【1】为文件名
        if fileType[1] == '.txt':  #若文件名的后缀为txt,则继续遍历循环，否则退出循环
            continue
        name = outer_path + '/' + folder+ '/' +file + ' ' + str(int(label)) + '\n'  #name 为文件路径和文件名+空格+label+换行
        listText.write(name)  #在创建的txt文件中写入name
    listText.close() #关闭txt文件
    print('down!')
    print('****************')


outer_path = 'E:\MaxImportant\learning\code&exe\DeepL'  # 这里是你的图片路径

folder = 'trainimages'
if __name__ == '__main__':  #主函数
    folderlist = os.listdir(outer_path)# 列举文件夹
    generate(os.path.join(outer_path, folder))#调用generate函数，函数中的参数为：（图片路径+文件夹名，标签号）
