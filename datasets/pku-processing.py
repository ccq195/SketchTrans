
import random
import glob
import re
import os
import os.path as osp

photo_path = '/omnisky3/public-datasets/ccq/PKUSketchRE-ID_V1/photo/'
sketch_path = '/omnisky3/public-datasets/ccq/PKUSketchRE-ID_V1/sketch/'
photo_imgs= glob.glob(osp.join(photo_path, '*.jpg'))
sketch_imgs= glob.glob(osp.join(sketch_path, '*.jpg'))

for t in range(10):
    train_visible = 'train_visible_{}'.format(t+1) + '.txt'
    train_sketch = 'train_sketch_{}'.format(t+1) + '.txt'
    test_visible = 'test_visible_{}'.format(t+1) + '.txt'
    test_sketch = 'test_sketch_{}'.format(t+1) + '.txt'

    path1 = open(train_visible,'a+')
    path2 = open(train_sketch,'a+')
    path3 = open(test_visible,'a+')
    path4 = open(test_sketch,'a+')
    ## a
    f = open('/omnisky3/public-datasets/ccq/PKUSketchRE-ID_V1/styleAnnotation/a_46.txt','r')
    f1 = f.readline()
    a = []
    while f1:
        a.append(int(f1))
        f1 = f.readline()
    f.close()
    a1 = random.sample(a,11)
    a2 = []
    for x in a:
        if a1.count(x)==0:
            a2.append(x)

    for r in a1:
        for img in photo_imgs:
            id = (img.split('/')[-1]).split('_')[0]
            camid = (img.split('/')[-1]).split('_')[1]
            if int(id)==r:
                path3.write('photo/'+img.split('/')[-1]+' '+ id + ' '+ camid + ' ' + '0' +'\n')
        for ske in sketch_imgs:
            id = (ske.split('/')[-1]).split('.')[0]
            camid = '0'
            # camid = (img.split('/')[-1]).split('_')[1]
            if int(id)==r:
                path4.write('sketch/'+ske.split('/')[-1]+' '+ id + ' '+ camid + ' ' + '1'+ '\n')

    for r in a2:
        for img in photo_imgs:
            id = (img.split('/')[-1]).split('_')[0]
            camid = (img.split('/')[-1]).split('_')[1]
            if int(id)==r:
                path1.write('photo/' + img.split('/')[-1] + ' ' + id + ' ' + camid + ' ' + '0' + '\n')
        for ske in sketch_imgs:
            id = (ske.split('/')[-1]).split('.')[0]
            camid = '0'
            # camid = (img.split('/')[-1]).split('_')[1]
            if int(id)==r:
                path2.write('sketch/' + ske.split('/')[-1] + ' ' + id + ' ' + camid + ' ' + '1' + '\n')



    #
    # for i in range(len(a1)):
    #     test.write(str(a1[i])+'\n')
    # for i in range(len(a2)):
    #     train.write(str(a2[i])+'\n')


    ## b
    f = open('/omnisky3/public-datasets/ccq/PKUSketchRE-ID_V1/styleAnnotation/b_20.txt','r')
    f1 = f.readline()
    a = []
    while f1:
        a.append(int(f1))
        f1 = f.readline()
    f.close()
    a1 = random.sample(a,5)
    a2 = []
    for x in a:
        if a1.count(x)==0:
            a2.append(x)

    for r in a1:
        for img in photo_imgs:
            id = (img.split('/')[-1]).split('_')[0]
            camid = (img.split('/')[-1]).split('_')[1]
            if int(id)==r:
                path3.write('photo/'+img.split('/')[-1]+' '+ id + ' '+ camid + ' ' + '0' + '\n')
        for ske in sketch_imgs:
            id = (ske.split('/')[-1]).split('.')[0]
            camid = '0'
            # camid = (img.split('/')[-1]).split('_')[1]
            if int(id)==r:
                path4.write('sketch/'+ske.split('/')[-1]+' '+ id + ' '+ camid + ' ' + '2' + '\n')

    for r in a2:
        for img in photo_imgs:
            id = (img.split('/')[-1]).split('_')[0]
            camid = (img.split('/')[-1]).split('_')[1]
            if int(id)==r:
                path1.write('photo/' + img.split('/')[-1] + ' ' + id + ' ' + camid + ' ' + '0'+ '\n')
        for ske in sketch_imgs:
            id = (ske.split('/')[-1]).split('.')[0]
            camid = '0'
            # camid = (img.split('/')[-1]).split('_')[1]
            if int(id)==r:
                path2.write('sketch/' + ske.split('/')[-1] + ' ' + id + ' ' + camid + ' ' + '2' + '\n')

    # for i in range(len(a1)):
    #     test.write(str(a1[i])+'\n')
    # for i in range(len(a2)):
    #     train.write(str(a2[i])+'\n')
    ## c
    f = open('/omnisky3/public-datasets/ccq/PKUSketchRE-ID_V1/styleAnnotation/c_79.txt','r')
    f1 = f.readline()
    a = []
    while f1:
        a.append(int(f1))
        f1 = f.readline()
    f.close()
    a1 = random.sample(a,20)
    a2 = []
    for x in a:
        if a1.count(x)==0:
            a2.append(x)

    for r in a1:
        for img in photo_imgs:
            id = (img.split('/')[-1]).split('_')[0]
            camid = (img.split('/')[-1]).split('_')[1]
            if int(id)==r:
                path3.write('photo/'+img.split('/')[-1]+' '+ id + ' '+ camid + ' ' + '0' + '\n')
        for ske in sketch_imgs:
            id = (ske.split('/')[-1]).split('.')[0]
            camid = '0'
            # camid = (img.split('/')[-1]).split('_')[1]
            if int(id)==r:
                path4.write('sketch/'+ske.split('/')[-1]+' '+ id + ' '+ camid + ' ' + '3' + '\n')

    for r in a2:
        for img in photo_imgs:
            id = (img.split('/')[-1]).split('_')[0]
            camid = (img.split('/')[-1]).split('_')[1]
            if int(id)==r:
                path1.write('photo/' + img.split('/')[-1] + ' ' + id + ' ' + camid + ' ' + '0' + '\n')
        for ske in sketch_imgs:
            id = (ske.split('/')[-1]).split('.')[0]
            camid = '0'
            # camid = (img.split('/')[-1]).split('_')[1]
            if int(id)==r:
                path2.write('sketch/' + ske.split('/')[-1] + ' ' + id + ' ' + camid + ' ' + '3' + '\n')
    # for i in range(len(a1)):
    #     test.write(str(a1[i])+'\n')
    # for i in range(len(a2)):
    #     train.write(str(a2[i])+'\n')
    ## d
    f = open('/omnisky3/public-datasets/ccq/PKUSketchRE-ID_V1/styleAnnotation/d_33.txt','r')
    f1 = f.readline()
    a = []
    while f1:
        a.append(int(f1))
        f1 = f.readline()
    f.close()
    a1 = random.sample(a,8)
    a2 = []
    for x in a:
        if a1.count(x)==0:
            a2.append(x)
    # for i in range(len(a1)):
    #     test.write(str(a1[i])+'\n')
    # for i in range(len(a2)):
    #     train.write(str(a2[i])+'\n')
    for r in a1:
        for img in photo_imgs:
            id = (img.split('/')[-1]).split('_')[0]
            camid = (img.split('/')[-1]).split('_')[1]
            if int(id)==r:
                path3.write('photo/'+img.split('/')[-1]+' '+ id + ' '+ camid + ' ' + '0'+ '\n')
        for ske in sketch_imgs:
            id = (ske.split('/')[-1]).split('.')[0]
            camid = '0'
            # camid = (img.split('/')[-1]).split('_')[1]
            if int(id)==r:
                path4.write('sketch/'+ske.split('/')[-1]+' '+ id + ' '+ camid + ' ' + '4'+ '\n')

    for r in a2:
        for img in photo_imgs:
            id = (img.split('/')[-1]).split('_')[0]
            camid = (img.split('/')[-1]).split('_')[1]
            if int(id)==r:
                path1.write('photo/' + img.split('/')[-1] + ' ' + id + ' ' + camid + ' ' + '0' + '\n')
        for ske in sketch_imgs:
            id = (ske.split('/')[-1]).split('.')[0]
            camid = '0'
            # camid = (img.split('/')[-1]).split('_')[1]
            if int(id)==r:
                path2.write('sketch/' + ske.split('/')[-1] + ' ' + id + ' ' + camid + ' ' + '4' + '\n')


    ##e
    f = open('/omnisky3/public-datasets/ccq/PKUSketchRE-ID_V1/styleAnnotation/e_22.txt','r')
    f1 = f.readline()
    a = []
    while f1:
        a.append(int(f1))
        f1 = f.readline()
    f.close()
    a1 = random.sample(a,6)
    a2 = []
    for x in a:
        if a1.count(x)==0:
            a2.append(x)
    # for i in range(len(a1)):
    #     test.write(str(a1[i])+'\n')
    # for i in range(len(a2)):
    #     train.write(str(a2[i])+'\n')
    for r in a1:
        for img in photo_imgs:
            id = (img.split('/')[-1]).split('_')[0]
            camid = (img.split('/')[-1]).split('_')[1]
            if int(id)==r:
                path3.write('photo/'+img.split('/')[-1]+' '+ id + ' '+ camid + ' ' + '0'+ '\n')
        for ske in sketch_imgs:
            id = (ske.split('/')[-1]).split('.')[0]
            camid = '0'
            # camid = (img.split('/')[-1]).split('_')[1]
            if int(id)==r:
                path4.write('sketch/'+ske.split('/')[-1]+' '+ id + ' '+ camid + ' ' + '4'+ '\n')

    for r in a2:
        for img in photo_imgs:
            id = (img.split('/')[-1]).split('_')[0]
            camid = (img.split('/')[-1]).split('_')[1]
            if int(id)==r:
                path1.write('photo/' + img.split('/')[-1] + ' ' + id + ' ' + camid + ' ' + '0'+ '\n')
        for ske in sketch_imgs:
            id = (ske.split('/')[-1]).split('.')[0]
            camid = '0'
            # camid = (img.split('/')[-1]).split('_')[1]
            if int(id)==r:
                path2.write('sketch/' + ske.split('/')[-1] + ' ' + id + ' ' + camid + ' ' + '4'+ '\n')

    path1.close()
    path2.close()
    path3.close()
    path4.close()
