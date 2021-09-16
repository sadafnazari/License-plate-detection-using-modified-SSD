# -*- coding: utf-8 -*-

import xml.etree.ElementTree as ET


def parse_object(xmlFile):
    Tree = ET.parse(xmlFile) 
    root = Tree.getroot()
    object_set = root.findall('object')
    
    object_list = []
    for one in object_set:
        
        # get class
        name = one.find('name')
        classes = name.text
        
        # get coordinates of the box 
        bndbox = one.findall('bndbox')
        
        x1= bndbox[0].find('x1')
        x1 = int(x1.text)

        y1= bndbox[0].find('y1')
        y1 = int(y1.text)

        x2 = bndbox[0].find('x2')
        x2 = int(x2.text)

        y2 = bndbox[0].find('y2')
        y2 = int(y2.text)

        x3 = bndbox[0].find('x3')
        x3 = int(x3.text)

        y3 = bndbox[0].find('y3')
        y3 = int(y3.text)

        x4 = bndbox[0].find('x4')
        x4 = int(x4.text)

        y4 = bndbox[0].find('y4')
        y4 = int(y4.text)

        
        # patch_info = {'classes':classes,'xmin':xmin,'ymin':ymin,'xmax':xmax,'ymax':ymax}
        patch_info = {'classes':classes, 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'x3': x3, 'y3': y3, 'x4': x4, 'y4': y4}
        object_list.append(patch_info)
        
    return object_list


def parse_size(xmlFile):
    Tree = ET.parse(xmlFile) 
    root = Tree.getroot()
    size = root.findall('size')[0]
    
    width = size.find('width')
    width = int(width.text)
    
    height = size.find('height')
    height = int(height.text)
    
    depth = size.find('depth')
    depth = int(depth.text)  
    
    size = {'width':width,'height':height,'depth':depth}
        
    return size
