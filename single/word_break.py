# -*- coding: utf-8 -*-
# @Time        : 2018/7/10 14:39
# @Author      : panxiaotong
# @Description : DSSM模型分词

import jieba
import sys
import re

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("word_break <self-defined dictionary> <stopword file> <input file> <output file>")
        sys.exit()

    jieba.load_userdict(sys.argv[1])
    stopword_dict = {}
    with open(sys.argv[2], 'r') as input_file:
        for line in input_file:
            line = line.replace('\r','').replace('\n','').strip()
            if line not in stopword_dict:
                stopword_dict[line] = 1
        input_file.close()
    print(len(stopword_dict))

    float_digit_pattern = re.compile(r"-?([1-9]\d*\.\d*|0\.\d*[1-9]\d*|0?\.0+|0)$")
    integ_digit_pattern = re.compile(r"-?[1-9]\d*")

    with open(sys.argv[4], 'w') as output_file:
        with open(sys.argv[3], 'r') as input_file:
            for line in input_file:    # <user_query>\001<document1>\t<label1>\002<document2>\t<label2>
                elements = line.replace('\r','').replace('\n', '').split('\001')
                if len(elements) < 2:
                    continue
                query = elements[0]
                word_list = jieba.cut(query, cut_all=False)
                word_list = [item.encode('utf-8') for item in word_list if
                             item.encode('utf-8') not in stopword_dict and
                             item.encode('utf-8').find(" ") == -1 and
                             float_digit_pattern.match(item.encode("utf-8")) == None and
                             integ_digit_pattern.match(item.encode("utf-8")) == None]
                if len(word_list) == 0:
                    continue
                query = (',').join([item for item in word_list])
                documents_list = elements[1].split('\002')
                flag = False
                doc_list = []
                for document in documents_list:
                    document_level = document.split("\t")
                    document = document_level[0]
                    word_list = jieba.cut(document, cut_all=False)
                    word_list = [item.encode('utf-8') for item in word_list if
                                 item.encode('utf-8') not in stopword_dict and
                                 item.encode('utf-8').find(" ") == -1 and
                                 float_digit_pattern.match(item.encode("utf-8")) == None and
                                 integ_digit_pattern.match(item.encode("utf-8")) == None]
                    if len(word_list) == 0:
                        flag = True
                        break
                    document = (',').join([item for item in word_list])
                    level = document_level[1]
                    doc_list.append(document+"\t"+level)
                if flag == True:
                    continue
                output_file.write(query + "\001" + ("\002").join(doc_list) + "\n")
            input_file.close()
        output_file.close()