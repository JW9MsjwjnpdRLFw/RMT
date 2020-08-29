import os

dir = ['trainA', 'trainB', 'testA', 'testB']

for d in dir:
    img_names = os.listdir(d)
    f = open('list_' + d + '.txt', "w")
    for img in img_names:
        f.write('./' + img + '\n')
    # print(img_names)