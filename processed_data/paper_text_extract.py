import numpy as np
from transformers import (BertForSequenceClassification, BertTokenizer, LlamaForSequenceClassification, LlamaTokenizer,
                          GemmaForSequenceClassification, GemmaTokenizer, AutoTokenizer)
import json
import pickle
import settings
import utils
from RLmethod.pstenv import extract_paper_text_info
import torch
from os.path import join
from tqdm import tqdm


save_dir = settings.PROJ_DIR + "/data_processed"
train_data = utils.load_json(save_dir, "Train_papers_refs_pair.json")
valid_data = utils.load_json(save_dir, "Valid_papers_refs_pair.json")
train_data_small = {key: train_data[key] for key in list(train_data.keys())[:4]}
valid_data_small = {key: valid_data[key] for key in list(valid_data.keys())[:4]}


def get_global_paper_text():

    num = 0
    global_paper_dic = dict()
    for key in tqdm(train_data.keys() | valid_data.keys()):
        query_paper = key    # key
        global_info = extract_paper_text_info(query_paper, query_paper=query_paper, context_pos=False, content=True)

        query_text_input = "[SEP]".join(global_info)

        if query_text_input is None or query_text_input == "":
            query_text_input = ["[SEP]"]
            num += 1

        global_paper_dic[query_paper] = query_text_input
    print("Number of empty global text: ", num)
    utils.dump_json(global_paper_dic, save_dir, "global_paper_text.json")


def get_local_paper_text():
    num = 0
    local_paper_dic = dict()
    # for key in tqdm(train_data_small.keys() | valid_data_small.keys()):      # train_data.keys() | valid_data.keys()):
    for key in tqdm(train_data.keys() | valid_data.keys()):
        local_paper_dic[key] = dict()
        query_paper = key
        try:
            pairs = train_data[key]
        except:
            pairs = valid_data[key]

        for item in pairs:
            local_info = extract_paper_text_info(item[0], query_paper=query_paper, context_pos=item[1], content=False)
            local_text_input = "[SEP]".join([str(tokens) if tokens is not None else "" for tokens in local_info])
            if local_text_input is None or local_text_input == "":
                local_text_input = ["[SEP]"]
                num += 1
            local_paper_dic[key][item[1]] = local_text_input
    print("Number of empty local text: ", num)
    utils.dump_json(local_paper_dic, save_dir, "local_paper_text.json")


if __name__ == "__main__":

    get_global_paper_text()
    get_local_paper_text()

