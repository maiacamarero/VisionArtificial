label = "default_label"
description = "default_description"

#------------------------------------------

def label_to_int(string_label):
    if string_label == 'circle':
        return 1
    if string_label == 'square':
        return 2
    if string_label == 'triangle':
        return 3
    else:
        raise Exception('unknown class_label')
    
#------------------------------------------

def int_to_label(int_label):
    if int_label == 1:
        return 'circle'
    elif int_label == 2:
        return 'square'
    elif int_label == 3:
        return 'triangle'
    else:
        raise Exception('unknown class_label')