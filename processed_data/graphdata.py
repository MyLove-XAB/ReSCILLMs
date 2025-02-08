from os.path import join
import json
import networkx as nx
import os
import utils
import settings
from tqdm import tqdm


# 设置线程数
os.environ['NUMEXPR_MAX_THREADS'] = '16'  # 设置线程数为16


def get_triple(data_dir, ref_data, dblp_data):
    # 创建一个空的有向图
    G = nx.DiGraph()

    # strID2intID
    strID2intID = {}        # {strID: intID}，strID是paper或者ref的strID，intID是图中节点的ID
    ett2intID = {}      # 不包含papers和refs

    relation2intID = {"cites": 0, "published_in": 1, "has_keyword": 2, "author_of": 3}     # 关系的ID
    with open(join(data_dir, "relation2intID.txt"), 'w', encoding='utf-8') as file:
        # file.write("Unknown" + '\t' + str(0) + '\n')
        for relation, intID in relation2intID.items():
            file.write(str(relation) + '\t' + str(intID) + '\n')

    tmpID = 1

    # 添加引用关系到图中
    for paper, references in ref_data.items():
        if strID2intID.get(paper) is None:
            strID2intID[paper] = tmpID
            tmpID += 1
        for refs in references:
            ref = refs[0] if refs[0] else paper+refs[1]               # 如果refs[0]为空，则取refs[1]: bID，否则取refs[0]: strID
            G.add_edge(paper, ref, relationship='cites')            # 这里包含了不在dblp_data中的paper和ref节点
                                                                    # 后续过程中，如果图结构缺失，则用0来表示
            if strID2intID.get(ref) is None:
                strID2intID[ref] = tmpID
                tmpID += 1

    # 添加ground_truth 到图中
    with open(join(save_dir, "ground_truth_bid.json"), 'r', encoding='utf-8') as file:
        ground_truth_bid = json.load(file)

    with open(join(save_dir, "ground_truth_index.json"), 'r', encoding='utf-8') as file:
        ground_truth_index = json.load(file)
        for paper, gt_refs in ground_truth_index.items():
            for i, gt_ref in enumerate(gt_refs):
                if gt_ref is None:
                    try:
                        gt_ref = ground_truth_bid[paper][i]
                        G.add_edge(paper, paper + gt_ref, relationship='cites')
                        if strID2intID.get(gt_ref) is None:
                            strID2intID[gt_ref] = tmpID
                            tmpID += 1
                        continue
                    except:
                        continue

                G.add_edge(paper, gt_ref, relationship='cites')     # some may be duplicated, but it doesn't matter
                if strID2intID.get(gt_ref) is None:
                    strID2intID[gt_ref] = tmpID
                    tmpID += 1

        for paper, gt_bid in ground_truth_bid.items():
            for gt_bid in gt_refs:
                if gt_bid is None:
                    continue
                G.add_edge(paper, paper+gt_bid, relationship='cites')     # some may be duplicated, but it doesn't matter
                if strID2intID.get(paper+gt_bid) is None:
                    strID2intID[paper+gt_bid] = tmpID
                    tmpID += 1

    # 添加详细信息到图中
    for paper, details in dblp_data.items():
        if paper is None:
            continue  # 如果 paper 为空，则跳过该项

        title = details.get('title', '').lower().strip("\"")        # 注意，全部转换为小写
        n_citation = details.get('n_citation', '')
        venue = details.get('venue', '')
        keywords = details.get('keywords', [])
        authors = details.get('authors', [])

        G.add_node(paper, title=title, n_citation=n_citation)       # title和n_citation是paper的属性，不是节点

        if venue:
            tmp_venue = venue.replace('\n', ' ').replace('\t', ' ').split()
            venue = " ".join(tmp_venue).lower()
            G.add_node(venue, type='venue')
            G.add_edge(paper, venue, relationship='published_in')
            if ett2intID.get(venue) is None:
                ett2intID[venue] = tmpID
                tmpID += 1

        for keyword in keywords:
            # 分割关键词并添加到图中
            for kw in keyword.split(','):
                kw = kw.strip().lower()  # 去除空格
                kw = kw.strip("\"").strip()     # 去掉前后的引号，防止出现为封闭的情况,然后再去一下空格
                if kw:
                    G.add_node(kw, type='keyword')
                    G.add_edge(paper, kw, relationship='has_keyword')
                    if ett2intID.get(kw) is None:
                        ett2intID[kw] = tmpID
                        tmpID += 1

        for author in authors:
            # 分割作者并添加到图中
            if author:
                auth = author.strip().lower()  # 去除前后空格
                if auth:
                    G.add_node(auth, type='author')
                    G.add_edge(auth, paper, relationship='author_of')
                    if ett2intID.get(auth) is None:
                        ett2intID[auth] = tmpID
                        tmpID += 1

    # 打印图的基本信息
    print("节点数:", G.number_of_nodes())
    print("边数:", G.number_of_edges())

    # 保存三元组到JSON文件
    triples = []
    for source, target, data in G.edges(data=True):
        relationship = data.get('relationship', 'cites')
        triples.append({"head": source, "relation": relationship, "tail": target})

    output_file = join(data_dir, "knowledge_graph_triples.json")
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(triples, file, indent=4, ensure_ascii=False)

    output_file_txt = join(data_dir, "knowledge_graph_triples.txt")
    with open(output_file_txt, 'w', encoding='utf-8') as file:      # txt不合适，因为有点内容里面包含了换行符和制表符
        for dic in triples:
            file.write(str(dic['head']))
            file.write('\t')
            file.write(str(dic['relation']))
            file.write('\t')
            file.write(str(dic['tail']))
            file.write('\n')
    print(f"三元组已保存到 {output_file}\n {output_file_txt}")

    output_file_txt = join(data_dir, "knowledge_graph_triples_ids.txt")
    with open(output_file_txt, 'w', encoding='utf-8') as file:  # txt不合适，因为有点内容里面包含了换行符和制表符
        for dic in triples:
            file.write(str(strID2intID.get(dic['head'], ett2intID.get(dic['head'], 0))))
            file.write('\t')
            file.write(str(relation2intID[dic['relation']]))            # relation 2 intID
            file.write('\t')
            # tmp = strID2intID.get(dic['tail'], ett2intID.get(dic['tail'], -1))
            file.write(str(strID2intID.get(dic['tail'], ett2intID.get(dic['tail'], 0))))
            file.write('\n')
    print(f"id编码的三元组已保存到 {output_file_txt}")

    with open(join(data_dir, "strID2intID.txt"), 'w', encoding='utf-8') as file:
        file.write("Unknown" + '\t' + str(0) + '\n')
        for strID, intID in strID2intID.items():
            file.write(str(strID) + '\t' + str(intID) + '\n')
    print(f"strID2intID已保存到 {join(data_dir, 'strID2intID.txt')}")

    with open(join(data_dir, "ett2intID.txt"), 'w', encoding='utf-8') as file:
        file.write("Unknown" + '\t' + str(0) + '\n')
        for ett, intID in ett2intID.items():
            file.write(str(ett) + '\t' + str(intID) + '\n')


if __name__ == "__main__":
    # 构建图谱需要所用的训练、验证、测试数据
    # data_dir = join(settings.DATA_TRACE_DIR, "PST")
    save_dir = join(settings.PROJ_DIR, "data_processed")
    # 读取JSON文件
    with open(join(save_dir, "Train_papers_refs_pair.json"), 'r', encoding='utf-8') as file:
        train_ref_data = json.load(file)
    with open(join(save_dir, "Valid_papers_refs_pair.json"), 'r', encoding='utf-8') as file:
        valid_ref_data = json.load(file)

    ref_data = {**train_ref_data, **valid_ref_data}

    with open(join(save_dir, "paper_info.json"), 'r', encoding='utf-8') as file:
        paper_info = json.load(file)

    # 生成知识图谱三元组
    get_triple(save_dir, ref_data, paper_info)      # 用paper_info代替dblp_data
