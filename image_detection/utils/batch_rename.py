import os


def batch_rename():
    file_dir = '/image_detection/data/org_jpg/'
    i = 0
    for file in os.listdir(file_dir):
        old_filename = os.path.join(file_dir, file)
        new_filename = os.path.join(file_dir, str('%05d' % i) + '.jpg')
        os.rename(old_filename, new_filename)
        i += 1


batch_rename()

