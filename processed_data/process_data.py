import json
from os.path import join
from tqdm import tqdm
from collections import defaultdict as dd
import logging
import settings
import utils
from bs4 import BeautifulSoup
from lxml import etree
from fuzzywuzzy import fuzz
import logging

# 配置日志记录
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')  # include timestamp


def is_fuzzy_match(title_text, title_set, threshold=80):
    """检查 title_text 是否与 title_set 中的任何标题模糊匹配"""
    for existing_title in title_set:
        if fuzz.ratio(title_text, existing_title) >= threshold:
            return True, existing_title
    return False, None


def get_paper_reference():      # valid json
    title_to_id = {}
    id_to_title = {}
    # 初始化一个计数器 parse_err_cnt，用于记录解析错误的总数
    parse_err_cnt = 0

    with open(join(data_dir, dblp_fname), "r", encoding="utf-8") as myFile:
        for i, line in enumerate(myFile):
            if len(line) <= 2:
                continue

            # 如果当前行数是 100000 的倍数，则输出当前行的论文数量和解析错误的总数
            if i % 100000 == 0:
                logging.info("reading papers %d, parse err cnt %d", i, parse_err_cnt)
            try:
                paper_tmp = json.loads(line.strip())

                # 将论文的引文信息添加到 paper_dict_open 字典中
                paper_dict_open[paper_tmp["id"]] = paper_tmp.get("references", [])
                # 将标题和 ID 添加到 title_to_id 字典中
                title_to_id[paper_tmp["title"].lower()] = paper_tmp["id"]
                # 将 ID 和标题添加到 id_to_title 字典中
                id_to_title[paper_tmp["id"]] = paper_tmp["title"]

            except Exception as e:
                parse_err_cnt += 1
                logging.error("Parsing error occurred: %s", str(e))

    # 从 papers_train 和 papers_valid 中获取论文 ID 对应的引文信息
    V_paper_id_dict = {}
    TR_paper_id_dict = {}
    # articles_with_more_xml_refs = []
    in_dir = join(data_dir, "paper-xml")
    strid_to_title = dict()

    for paper in tqdm(papers_train):        # train json data
        label_id_set = set([item["_id"] for item in paper["refs_trace"]])
        pid = paper["_id"]
        gt_ = paper["refs_trace"][0]["_id"]     # ground truth
        try:
            strid_to_title[pid] = id_to_title[pid].lower()
        except KeyError:
            print("train paper_id not found in dblp: ", pid)
        cur_refs = paper.get("references", [])
        for tmp_ref in cur_refs:
            try:
                strid_to_title[tmp_ref] = id_to_title[tmp_ref].lower()
            except KeyError:
                print("train ref_id not found in dblp: ", tmp_ref)
                print("train paper_id: ", pid)

        if len(cur_refs) == 0:
            continue

        refs_open = paper_dict_open.get(pid, [])            # list of references from dblp
        refs_update = list(set(cur_refs + refs_open))
        id_set = set(refs_update)
        title_set = {id_to_title[ref_id].lower() for ref_id in refs_update if ref_id in id_to_title}
        # TV_paper_id_dict[pid] = refs_update
        final_refs = []     # 用来存最终的结果

        f = open(join(in_dir, pid + ".xml"), encoding='utf-8')
        xml = f.read()
        bs = BeautifulSoup(xml, "xml")
        references = bs.find_all("biblStruct")
        bid_to_title = {}
        n_refs = 0
        for ref in references:
            if "xml:id" not in ref.attrs:
                continue
            bid = ref.attrs["xml:id"]
            if ref.analytic is None:
                continue
            if ref.analytic.title is None:
                continue
            if ref.analytic.title.text.lower() != "":
                bid_to_title[bid] = ref.analytic.title.text.lower()
            else:
                bid_to_title[bid] = ref.monogr.title.text.lower()

            b_idx = int(bid[1:]) + 1
            if b_idx > n_refs:
                n_refs = b_idx

        bib_sorted = ["b" + str(ii) for ii in range(n_refs)]  # 对bib做了排序

        for i in range(n_refs):
            title_text = bid_to_title.get(bib_sorted[i], "")
            if title_text == "":
                print("current train paper: ", pid)
                print("bib: ", bib_sorted[i])
                print("not found bid: ", i)
                # final_refs.append(bib_sorted[i])            # 没有title的话，就直接用bid
                final_refs.append((None, bib_sorted[i], 0))        # 没有title的话，就(None, bid, 0)
                continue
            signal, existing_title = is_fuzzy_match(title_text, title_set)
            strid = title_to_id[existing_title] if signal else None     # 如果匹配上了，就用strid，否则就是None
            if strid in label_id_set:
                label = 1
            else:
                label = 0
            if not signal:  # 在dblp中匹配不上
                # final_refs.append(title_text)
                print("train title_text: {} ```not in dblp".format(title_text))
                if label == 1:
                    final_refs.append((gt_, bib_sorted[i], label))          # 是ground truth的话，就(gt_, bid, label)
                else:
                    final_refs.append((None, bib_sorted[i], label))         # 有title但是title匹配不上，就(None, bid, label)
                # continue
            else:
                final_refs.append((title_to_id[existing_title], bib_sorted[i], label))     # (strid, bid, label)

        TR_paper_id_dict[pid] = final_refs

    for paper in tqdm(papers_valid):        # valid json data
        pid = paper["_id"]
        gt_ = ground_truth[pid]     # valid data中的ground_truth
        try:
            strid_to_title[pid] = id_to_title[pid].lower()
        except KeyError:
            print("paper_id not found in dblp: ", pid)
        cur_refs = paper.get("references", [])              # list of references from valid data
        cur_refs_title_to_id = dict()
        for tmp_ref in cur_refs:
            try:
                cur_refs_title_to_id[id_to_title[tmp_ref].lower()] = tmp_ref
                strid_to_title[tmp_ref] = id_to_title[tmp_ref].lower()
            except KeyError:
                # exist some ref_id can not find in dblp: v15.1: 5e8d8e6d9fced0a24b5d669e  v15: 62376b725aee126c0f0a7412
                print("ref_id not found in dblp: ", tmp_ref)
                print("paper_id: ", pid)
        if len(cur_refs) == 0:
            continue

        refs_open = paper_dict_open.get(pid, [])            # list of references from dblp
        refs_update = list(set(cur_refs + refs_open))
        id_set = set(refs_update)
        title_set = {id_to_title[ref_id].lower() for ref_id in refs_update if ref_id in id_to_title}
        # TV_paper_id_dict[pid] = refs_update
        final_refs = []     # 用来存最终的结果

        f = open(join(in_dir, pid + ".xml"), encoding='utf-8')
        xml = f.read()
        bs = BeautifulSoup(xml, "xml")
        references = bs.find_all("biblStruct")
        bid_to_title = {}
        n_refs = 0
        for ref in references:
            if "xml:id" not in ref.attrs:
                continue
            bid = ref.attrs["xml:id"]
            if ref.analytic is None:
                continue
            if ref.analytic.title is None:
                continue
            if ref.analytic.title.text.lower() != "":
                bid_to_title[bid] = ref.analytic.title.text.lower()
            else:
                bid_to_title[bid] = ref.monogr.title.text.lower()

            b_idx = int(bid[1:]) + 1
            if b_idx > n_refs:
                n_refs = b_idx

        assert len(sub_example_dict[pid]) == n_refs     # 确保reference的数量和submission_example中的数量一致，所以他通过这种方式和valid data中的数据匹配起来了，没有通过ref xml的id进行匹配的
        assert len(gt_) == n_refs
        bib_sorted = ["b" + str(ii) for ii in range(n_refs)]  # 对bib做了排序

        for i in range(n_refs):
            title_text = bid_to_title.get(bib_sorted[i], "")
            if title_text == "":
                final_refs.append((None, bib_sorted[i], gt_[i]))        # without title: (None, bid, label)
                continue
            signal, existing_title = is_fuzzy_match(title_text, title_set)
            if not signal:  # 在dblp中匹配不上
                final_refs.append((None, bib_sorted[i], gt_[i]))        # 有title但是title匹配不上，就(None, bid, label)
            else:
                final_refs.append((title_to_id[existing_title], bib_sorted[i], gt_[i]))     # (strid, bid, label)

        V_paper_id_dict[pid] = final_refs

    utils.dump_json(strid_to_title, save_dir, "strID_to_title.json")        # NCF中的重点
    utils.dump_json(V_paper_id_dict, save_dir, "Valid_papers_refs_pair.json")
    utils.dump_json(TR_paper_id_dict, save_dir, "Train_papers_refs_pair.json")

    print("训练集总数：", len(TR_paper_id_dict))
    print("验证集总数：", len(V_paper_id_dict))


def write_ground_truth():
    train_pair = utils.load_json(save_dir, "Train_papers_refs_pair.json")
    valid_pair = utils.load_json(save_dir, "Valid_papers_refs_pair.json")
    ground_truth_label = {}
    ground_truth_index = {}
    ground_truth_bid = {}

    for paper in tqdm(papers_train):        # train json data

        pid = paper["_id"]
        ground_truth_index[pid] = list()
        for gt in paper["refs_trace"]:
            ground_truth_index[pid].append(gt["_id"])

    for pid, refs in train_pair.items():
        ground_truth_label[pid] = list()
        ground_truth_bid[pid] = list()
        for ref in refs:
            # append label
            ground_truth_label[pid].append(ref[2])     # {pid: [label1, label2, ...]}, 0 or 1

            # append positive id
            if ref[2] == 1:         # 存在某个ref[2]都不为1的情况
                ground_truth_bid[pid].append(ref[1])      # {pid: [bid1, bid2, ...]}, "b1", "b2", ...

    for pid, refs in valid_pair.items():
        ground_truth_label[pid] = list()
        ground_truth_index[pid] = list()
        ground_truth_bid[pid] = list()
        for ref in refs:
            ground_truth_label[pid].append(ref[2])
            if ref[2] == 1:
                ground_truth_index[pid].append(ref[0])
                ground_truth_bid[pid].append(ref[1])

    utils.dump_json(ground_truth_label, save_dir, "ground_truth_label.json")
    utils.dump_json(ground_truth_index, save_dir, "ground_truth_index.json")
    utils.dump_json(ground_truth_bid, save_dir, "ground_truth_bid.json")
    print("finished writing ground truth")


def load_dblp_data(data_dir, dblp_fname):
    parse_err_cnt = 0
    paper_dict_dblp = {}

    with open(join(data_dir, dblp_fname), "r", encoding="utf-8") as myFile:
        for i, line in enumerate(myFile):
            if len(line) <= 2:
                continue
            if i % 10000 == 0:
                logger.info("reading papers %d, parse err cnt %d", i, parse_err_cnt)

            try:
                paper_tmp = json.loads(line.strip())
                paper_dict_dblp[paper_tmp["id"]] = paper_tmp
            except Exception as e:
                logger.error(f"Error parsing line {i}: {e}")
                parse_err_cnt += 1

    logger.info("number of papers after loading %d", len(paper_dict_dblp))
    return paper_dict_dblp


def get_paper_info():
    # DBLP takes precedence, followed by xml
    paper_more_info = dd(dict)
    paper_dict_dblp = load_dblp_data(data_dir, "DBLP-Citation-network-V15.1.json")
    for paper in tqdm(papers_train + papers_valid):
        cur_pid = paper["_id"]
        ref_ids = paper.get("references", [])
        pids = [cur_pid] + ref_ids      # including paper id and reference ids
        for pid in pids:
            # extract dblp info
            if pid in paper_more_info.keys():   # 如果已经在paper_more_info中
                # confirm whether the information is complete
                continue

            if pid in paper_dict_dblp:      # 如果在dblp中找不到这个pid，就跳过
                cur_paper_info = paper_dict_dblp[pid]
                title = cur_paper_info.get("title", "")
                abstract = cur_paper_info.get("abstract", "")
                keywords = cur_paper_info.get("keywords", "")
                cur_authors = [a.get("name", "") for a in cur_paper_info.get("authors", [])]
                n_citation = cur_paper_info.get("n_citation", 0)
                venue = cur_paper_info.get("venue", "")

            else:               # 如果在dblp中找不到这个pid，就先用空白填充，然后去xml中找
                title = ""
                abstract = ""
                keywords = ""
                cur_authors = []
                n_citation = 0
                venue = ""

            # extract xml info, whether the pid in dblp or not
            try:
                path = join(in_dir, pid + ".xml")
                tree = etree.parse(path)
                root = tree.getroot()
                listBibl = root.xpath("//*[local-name()='listBibl']")[0]
                biblStruct = listBibl.getchildren()
                num_ref_xml = len(biblStruct)

                with open(path, 'r', encoding='utf-8') as f:
                    xml_content = f.read()
                bs = BeautifulSoup(xml_content, 'xml')
                if not title:
                    title_tag = bs.fileDesc.titleStmt.title
                    title = title_tag.text if title_tag else ""
                if not abstract:
                    abstract = bs.profileDesc.abstract.text.strip() if bs.profileDesc.abstract else ""

                body_tag = bs.find('body')
                div_content_list = []
                if body_tag:
                    # 在<body>标签内部查找所有的<div>标签
                    div_tags = body_tag.find_all('div')
                    # 初始化一个空列表，用于存储提取的内容
                    div_content_list = []
                    # 遍历每个<div>标签，提取其文本内容，并添加到列表中
                    for div_tag in div_tags:
                        div_content_list.append(div_tag.get_text())
                # 将列表中的内容连接成一个字符串
                div_content = '\n'.join(div_content_list)

            except OSError:
                num_ref_xml = 0
                div_content = ""
                print('not exits xml ' + pid)

            paper_more_info[pid] = {
                "title": title, "abstract": abstract, "keywords": keywords,
                "authors": cur_authors, "n_citation": n_citation, "venue": venue,
                "num_ref": num_ref_xml, "content": div_content
            }

    print("number of papers after filtering", len(paper_more_info))
    utils.dump_json(paper_more_info, save_dir, "paper_info.json")


def bib_to_contexts():

    papers = utils.load_json(data_dir, "paper_source_trace_valid_wo_ans.json")
    papers_train = utils.load_json(data_dir, "paper_source_trace_train_ans.json")

    bib_to_contexts_dic = dd(dict)
    for paper in tqdm(papers + papers_train):
        cur_pid = paper["_id"]
        f = open(join(in_dir, cur_pid + ".xml"), encoding='utf-8')
        xml = f.read()
        bs = BeautifulSoup(xml, "xml")

        references = bs.find_all("biblStruct")
        bid_to_title = {}
        n_refs = 0
        for ref in references:
            if "xml:id" not in ref.attrs:
                continue
            bid = ref.attrs["xml:id"]
            if ref.analytic is None:
                continue
            if ref.analytic.title is None:
                continue
            bid_to_title[bid] = ref.analytic.title.text.lower()
            b_idx = int(bid[1:]) + 1
            if b_idx > n_refs:
                n_refs = b_idx

        bib_to_contexts = utils.find_bib_context(xml)
        bib_sorted = ["b" + str(ii) for ii in range(n_refs)]  # 对bib做了排序

        for bib in bib_sorted:
            cur_context = " ".join(bib_to_contexts[bib])
            bib_to_contexts_dic[cur_pid][bib] = (cur_context)

    print("len(bib_to_contexts_labels)", len(bib_to_contexts_dic))
    json.dump(bib_to_contexts_dic, open(join(save_dir, "bib_to_contexts.json"), "w", encoding="utf-8"))


if __name__ == "__main__":
    data_dir = join(settings.DATA_TRACE_DIR, "PST")
    save_dir = join(settings.PROJ_DIR, "data_processed")
    valid_dir = join(settings.DATA_TRACE_DIR, "PST-valid")
    in_dir = join(data_dir, "paper-xml")
    dblp_fname = "DBLP-Citation-network-V15.1.json"

    paper_dict_open = {}
    papers_train = utils.load_json(data_dir, "paper_source_trace_train_ans.json")
    papers_valid = utils.load_json(data_dir, "paper_source_trace_valid_wo_ans.json")
    sub_example_dict = utils.load_json(data_dir, "submission_example_valid.json")
    ground_truth = utils.load_json(valid_dir, "ground_truths_valid.json")

    get_paper_reference()
    write_ground_truth()
    bib_to_contexts()
    get_paper_info()

