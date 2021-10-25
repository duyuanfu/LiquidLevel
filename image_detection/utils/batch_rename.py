import os


def batch_rename():
    file_dir = '../data/org_png/'
    i = 0
    for file in os.listdir(file_dir):
        old_filename = os.path.join(file_dir, file)
        new_filename = os.path.join(file_dir, str('%05d' % i) + '.png')
        os.rename(old_filename, new_filename)
        i += 1


batch_rename()

