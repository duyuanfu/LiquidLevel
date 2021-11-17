import xml.etree.ElementTree as ET
import os


def read_xml_annotation(root, image_id):
    """
    :param root: XML_DIR
    :param image_id: filename of xml
    :return:
    """
    in_file = open(os.path.join(root, image_id), encoding='utf-8')
    tree = ET.parse(in_file)
    root = tree.getroot()
    bndboxlist = []

    for object in root.findall('object'):  # 找到root节点下的所有country节点
        bndbox = object.find('bndbox')  # 子节点下节点rank的值

        xmin = int(bndbox.find('xmin').text)
        xmax = int(bndbox.find('xmax').text)
        ymin = int(bndbox.find('ymin').text)
        ymax = int(bndbox.find('ymax').text)
        # print(xmin,ymin,xmax,ymax)
        bndboxlist.append([xmin, ymin, xmax, ymax])
        # print(bndboxlist)

    bndbox = root.find('object').find('bndbox')
    return bndboxlist


if __name__ == '__main__':
    XML_DIR = '../voc_hole/XML/'
    TXT_DIR = '../voc_hole/TXT/'

    for root, sub_folders, files in os.walk(XML_DIR):
        for name in files:
            bndbox = read_xml_annotation(XML_DIR, name)
            with open(os.path.join(TXT_DIR, name[:-4] + '.txt'), 'w') as f:
                f.write('1 ' + (str(*bndbox)[1: -1]).replace(',', ''))
                f.close()
