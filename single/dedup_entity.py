# -*- coding: utf-8 -*-
# @Time        : 2018/7/22 12:43
# @Author      : panxiaotong
# @Description : dedup entities

import sys

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("dedup_entity.py <input file> <output file>")
        sys.exit()

    dedup_dict = {}
    with open(sys.argv[2], 'w') as output_file:
        with open(sys.argv[1], 'r') as input_file:
            for line in input_file:
                line = line.replace('\r','').replace('\n','').strip()
                if line not in dedup_dict:
                    dedup_dict[line] = 1
            input_file.close()
        for k,v in dedup_dict.items():
            output_file.write(k + "\n")
        output_file.close()