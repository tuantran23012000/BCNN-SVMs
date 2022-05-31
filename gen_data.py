from __future__ import print_function, division

def convert_aircraft(root):
    #root = '/content/drive/MyDrive/Colab Notebooks/FPT-AI/data/fgvc-aircraft-2013b/data'
    train_txt = root + '/images_variant_trainval.txt'
    test_txt = root + '/images_variant_test.txt'
    variant_txt = root + '/variants.txt'

    ###
    variants_dict = {}
    with open(variant_txt,'r') as f:
        lines = f.readlines()
    index = 0
    for line in lines:
        variant = line.strip()
        if variant in variants_dict:
            continue
        else:
            variants_dict[variant] = index
            index += 1
    # print(index)

    ###
    train_lst = root + '/aircraft_train.txt'
    test_lst = root + '/aircraft_test.txt'

    train_f = open(train_lst,'a')
    with open(train_txt,'r') as f:
        lines = f.readlines()
    for line in lines:
        # print(line)
        lst= line.strip().split(' ',1)
        # print(lst)
        name,label = lst
        name = name +'.jpg'
        label = variants_dict[label]
        train_f.write('%s %d\n'%(name,label))
    train_f.close()
    test_f = open(test_lst,'a')
    with open(test_txt,'r') as f:
        lines = f.readlines()
    for line in lines:
        name,label = line.strip().split(' ',1)
        name = name+'.jpg'
        label = variants_dict[label]
        test_f.write('%s %d\n'%(name,label))
    test_f.close()
root = './data/fgvc-aircraft-2013b/data'
convert_aircraft(root)