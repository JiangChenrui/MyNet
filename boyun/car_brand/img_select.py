# -*- coding:utf-8 -*-
from pyexcel_xls import get_data
from pyexcel_xls import save_data
import glob
import sys
import os


# 读取xlsx表格，使用python3较为简便
xlsx_paths = "../车辆属性标注表格"
xlsx_path = []
for path in glob.glob(os.path.join(xlsx_paths, '*.xlsx')):
    xlsx_path.append(path)

label_txt = []
num = 0
for path in xlsx_path:
    xlsx = get_data(path)
    for key in xlsx.keys():
        sheet = xlsx[key]
        for i in range(1, len(sheet)):
            num += 1
            label_txt.append(str(sheet[i][1]) + '\t' + str(sheet[i][5]))

car_brand_txt = 'brand_txt.txt'
pf = open(car_brand_txt, 'w')

for label in label_txt:
    pf.write(label)
    pf.write('\n')
pf.close()
