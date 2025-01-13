import json
import os
import re
import subprocess
import time
import torch
import argparse
import GPUtil

from transformers import RobertaTokenizer, RobertaForMaskedLM, logging
from generic_templates import *
from matching_templates import *
from tool.logger import Logger
from tool.fault_localization import get_location
from tool.d4j import build_d4j1_2
from validate_patches import GVpatches
from bert_beam_search import BeamSearch

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def get_gpu_model():
    try:
        gpus = GPUtil.getGPUs()
        if not gpus:
            return "Not Found!"
        gpu_model = gpus[0].name
        return gpu_model
    except Exception as e:
        return f"Error：{e}"


def comment_remover(text):
    def replacer(match):
        s = match.group(0)
        if s.startswith('/'):
            return " "
        else:
            return s

    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE
    )
    return re.sub(pattern, replacer, text)


def add_new_line(file, line_loc, tokenizer, model, beam_width, re_rank=True, top_n_patches=-1):
    with open(file, 'r', encoding='utf-8', errors='ignore') as f:
        data = f.readlines()

    ret_before = []
    mask_token = "<mask>"
    pre_code = data[:line_loc]
    post_code = data[line_loc:]
    old_code = data[line_loc].strip()
    masked_line = " " + mask_token * 20 + " "
    line_size = 100
    while (1):
        pre_code_input = "</s> " + " ".join(
            [x.strip() for x in pre_code[-line_size:]])
        post_code_input = " ".join([x.strip() for x in post_code[0:line_size]]).replace("\n", "").strip()
        if tokenizer(pre_code_input + masked_line + post_code_input, return_tensors='pt')['input_ids'].size()[
            1] < 480:
            break
        line_size -= 1

    print(">>>>> Begin Some Very Long Beam Generation <<<<<")
    print("Context Line Size: {}".format(line_size))
    print("Context Before:\n{}".format(pre_code_input))
    print("Context After:\n{}".format(post_code_input))

    for token_len in range(1, 30):

        masked_line = " " + mask_token * token_len + " "
        beam_engine = BeamSearch(model, tokenizer, pre_code_input + masked_line + post_code_input, device,
                                 beam_width=beam_width, re_rank=re_rank)
        beam_list, masked_index = beam_engine.generate_beam()
        for beam in beam_list:
            ret_before.append(("".join(beam[2]), beam[0] / token_len, "Before " + masked_line))
    ret_before.sort(key=lambda x: x[1], reverse=True)
    ret_before = remove_redudant(ret_before)

    ret = []
    ret.extend(ret_before)
    ret.sort(key=lambda x: x[1], reverse=True)

    if top_n_patches == -1:
        return pre_code, old_code, ret, post_code
    else:
        return pre_code, old_code, ret[:top_n_patches], post_code


def process_file(file, line_loc, tokenizer, model, beam_width, re_rank=True, top_n_patches=-1):
    with open(file, 'r', encoding='utf-8', errors='ignore') as f:
        data = f.readlines()

    ret = []
    mask_token = "<mask>"
    pre_code = data[:line_loc]
    fault_line = comment_remover(data[line_loc].strip())
    old_code = data[line_loc].strip()
    post_code = data[line_loc + 1:]

    line_size = 100
    while (1):
        pre_code_input = "</s> " + " ".join([x.strip() for x in pre_code[-line_size:]])
        post_code_input = " ".join([x.strip() for x in post_code[0:line_size]]).replace("\n", "").strip()
        if tokenizer(pre_code_input + fault_line + post_code_input, return_tensors='pt')['input_ids'].size()[
            1] < 480:
            break
        line_size -= 1

    print(">>>>> Begin Some Very Long Beam Generation <<<<<")
    print("Context Line Size: {}".format(line_size))
    print("Context Before:\n{}".format(pre_code_input))
    print(">> {} <<".format(fault_line))
    print("Context After:\n{}".format(post_code_input))

    fault_line_token_size = tokenizer(fault_line, return_tensors='pt')["input_ids"].shape[1] - 2

    # MT6
    for rep in mutate_common_word(fault_line):
        ret.append((rep, 0, 'MT6: common replacement'))

    # MT7
    mt7_template = match_simple_operator(fault_line)
    for template in mt7_template:
        token_len = template.count("<mask>")
        masked_line = " " + template + " "
        beam_engine = BeamSearch(model, tokenizer, pre_code_input + masked_line + post_code_input, device,
                                 beam_width=beam_width, re_rank=re_rank)
        beam_list, masked_index = beam_engine.generate_beam()
        for beam in beam_list:
            index = 0
            gen_line = ""
            for c in masked_line.split(mask_token)[:-1]:
                gen_line += c
                gen_line += beam[2][index]
                index += 1
            gen_line += masked_line.split(mask_token)[-1]
            gen_line = gen_line[1:-1]
            ret.append((gen_line, beam[0] / token_len / 1000, 'MT7: ' + masked_line))

    # MT8.1
    mt81_templates = match_conditional_expression1(fault_line)
    for partial_beginning, partial_end in mt81_templates:
        for token_len in range(1, fault_line_token_size):
            masked_line = " " + partial_beginning + mask_token * token_len + partial_end + " "
            beam_engine = BeamSearch(model, tokenizer, pre_code_input + masked_line + post_code_input, device,
                                     beam_width=beam_width, re_rank=re_rank)
            beam_list, masked_index = beam_engine.generate_beam()
            for beam in beam_list:
                ret.append((partial_beginning + "".join(beam[2]) + partial_end, beam[0] / token_len / 10,
                            'MT8: ' + masked_line))

    # MT8.3-4
    mt834_templates = match_conditional_expression2(fault_line)
    for partial_beginning, partial_end in mt834_templates:
        for token_len in range(1, 11):
            masked_line = " " + partial_beginning + mask_token * token_len + partial_end + " "
            beam_engine = BeamSearch(model, tokenizer, pre_code_input + masked_line + post_code_input, device,
                                     beam_width=beam_width, re_rank=re_rank)
            beam_list, masked_index = beam_engine.generate_beam()
            for beam in beam_list:
                ret.append((partial_beginning + "".join(beam[2]) + partial_end, beam[0] / token_len / 10,
                            'MT8: ' + masked_line))

    # MT9
    mt9_templates = match_return_expression(fault_line)
    for partial_beginning, partial_end in mt9_templates:
        for token_len in range(1, 6):
            masked_line = " " + partial_beginning + mask_token * token_len + partial_end + " "
            beam_engine = BeamSearch(model, tokenizer, pre_code_input + masked_line + post_code_input, device,
                                     beam_width=beam_width, re_rank=re_rank)
            beam_list, masked_index = beam_engine.generate_beam()
            for beam in beam_list:
                ret.append((partial_beginning + "".join(beam[2]) + partial_end, beam[0] / token_len / 10,
                            'MT9: ' + masked_line))

    # MT10
    m10_templates = mutate_method_expression(fault_line, tokenizer)
    for match, length in m10_templates:
        for token_len in range(1, length + 5):
            if len(match.split(mask_token)) == 2:
                masked_line = " " + match.split(mask_token)[0] + mask_token * token_len + match.split(mask_token)[
                    1] + " "
                beam_engine = BeamSearch(model, tokenizer, pre_code_input + masked_line + post_code_input, device,
                                         beam_width=beam_width, re_rank=re_rank)
                beam_list, masked_index = beam_engine.generate_beam()
                for beam in beam_list:
                    ret.append((match.split(mask_token)[0] + "".join(beam[2]) + match.split(mask_token)[1],
                                beam[0] / token_len / 10, 'MT10: ' + masked_line))
            else:
                masked_line = " "
                masked_line += (mask_token * token_len).join(match.split(mask_token)) + " "
                beam_engine = BeamSearch(model, tokenizer, pre_code_input + masked_line + post_code_input, device,
                                         beam_width=beam_width, re_rank=re_rank)
                beam_list, masked_index = beam_engine.generate_beam()
                for beam in beam_list:
                    index = 0
                    gen_line = ""
                    for c in masked_line.split(mask_token)[:-1]:
                        gen_line += c
                        gen_line += beam[2][index]
                        index += 1
                    gen_line += masked_line.split(mask_token)[-1]
                    gen_line = gen_line[1:-1]
                    ret.append(
                        (gen_line, beam[0] / (token_len * (len(match.split(mask_token)) - 1)), 'MT10: ' + masked_line))

    # MT1
    mt1_template = insert_condition_checker(fault_line)
    for tbef, taft in mt1_template:
        for token_len in range(1, 11):
            masked_line = " " + tbef + mask_token * token_len + taft + " "
            beam_engine = BeamSearch(model, tokenizer, pre_code_input + masked_line + post_code_input, device,
                                     beam_width=beam_width, re_rank=re_rank)
            beam_list, masked_index = beam_engine.generate_beam()
            for beam in beam_list:
                ret.append((tbef + "".join(beam[2]) + taft, beam[0] / token_len, 'MT1: ' + masked_line))
    # MT2.1
    for token_len in range(2, fault_line_token_size + 5):
        masked_line = " " + mask_token * token_len + " " + fault_line + " "
        beam_engine = BeamSearch(model, tokenizer, pre_code_input + masked_line + post_code_input, device,
                                 beam_width=beam_width, re_rank=re_rank)
        beam_list, masked_index = beam_engine.generate_beam()
        for beam in beam_list:
            ret.append(("".join(beam[2]) + " " + fault_line, beam[0] / token_len, 'MT2: ' + masked_line))

    # MT2.2
    for token_len in range(2, fault_line_token_size + 5):
        masked_line = " " + fault_line + " " + mask_token * token_len + " "
        beam_engine = BeamSearch(model, tokenizer, pre_code_input + masked_line + post_code_input, device,
                                 beam_width=beam_width, re_rank=re_rank)
        beam_list, masked_index = beam_engine.generate_beam()
        for beam in beam_list:
            ret.append((fault_line + " " + "".join(beam[2]), beam[0] / token_len, 'MT2: ' + masked_line))

    # MT3
    ret.append((" ", 0, "MT3: Remove Buggy Statement"))

    # MT4
    for num in range(1, 11):
        ret.append((fault_line, -0.01, f"MT4: Move Statement Position {num}"))

    if fault_line_token_size <= 30:

        # MT5.1
        for token_len in range(fault_line_token_size - 5, fault_line_token_size + 5):
            if token_len <= 0:
                continue
            masked_line = " " + mask_token * token_len + " "
            beam_engine = BeamSearch(model, tokenizer, pre_code_input + masked_line + post_code_input, device,
                                     beam_width=beam_width, re_rank=re_rank)
            beam_list, masked_index = beam_engine.generate_beam()
            for beam in beam_list:
                ret.append(("".join(beam[2]), beam[0] / token_len, 'MT5: ' + masked_line))

        # MT5.2-3
        mt523_templates = mutate_line_expression1(fault_line)
        for partial_beginning, partial_end in mt523_templates:
            for token_len in range(2, fault_line_token_size):
                masked_line = " " + partial_beginning + mask_token * token_len + partial_end + " "
                beam_engine = BeamSearch(model, tokenizer, pre_code_input + masked_line + post_code_input, device,
                                         beam_width=beam_width, re_rank=re_rank)
                beam_list, masked_index = beam_engine.generate_beam()
                for beam in beam_list:
                    ret.append((partial_beginning + "".join(beam[2]) + partial_end, beam[0] / token_len,
                                'MT5: ' + masked_line))

    # MT5.4
    mt54_templates = mutate_line_expression2(fault_line)
    for partial_beginning, partial_end in mt54_templates:
        for token_len in range(1, 6):
            masked_line = " " + partial_beginning + mask_token * token_len + partial_end + " "
            beam_engine = BeamSearch(model, tokenizer, pre_code_input + masked_line + post_code_input, device,
                                     beam_width=beam_width, re_rank=re_rank)
            beam_list, masked_index = beam_engine.generate_beam()
            for beam in beam_list:
                ret.append(
                    (partial_beginning + "".join(beam[2]) + partial_end, beam[0] / token_len, 'MT5: ' + masked_line))

    ret.sort(key=lambda x: x[1], reverse=True)
    ret = remove_redudant(ret)

    if top_n_patches == -1:
        return pre_code, old_code, ret, post_code
    else:
        return pre_code, old_code, ret[:top_n_patches], post_code


def main(bug_ids, output_folder, skip_validation, beam_width, re_rank, perfect, top_n_patches):
    if bug_ids[0] == 'all':
        bug_ids = build_d4j1_2()


    model = RobertaForMaskedLM.from_pretrained("microsoft/codebert-base-mlm").to(device)
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base-mlm")

    for bug_id in bug_ids:
        subprocess.run('rm -rf ' + '/tmp/' + bug_id, shell=True)
        subprocess.run("defects4j checkout -p %s -v %s -w %s" % (
            bug_id.split('-')[0], bug_id.split('-')[1] + 'b', ('/tmp/' + bug_id)), shell=True)
        patch_pool_folder = "patches-pool"
        location = get_location(bug_id, perfect=perfect)
        if perfect:
            patch_pool_folder = "pfl-patches-pool-temp"

        testmethods = os.popen('defects4j export -w %s -p tests.trigger' % ('/tmp/' + bug_id)).readlines()

        logger = Logger(output_folder + '/' + bug_id + "_result.txt")
        logger.logo(args)
        gpu_model = get_gpu_model()
        logger.logo(f"GPU: {gpu_model}")
        validator = GVpatches(bug_id, testmethods, logger, patch_pool_folder=patch_pool_folder,
                                  skip_validation=skip_validation)

        file = location[0][0]
        line_number = location[0][1]
        print('Location: {} line # {}'.format(file, line_number))
        file = '/tmp/' + bug_id + '/' + file

        start_time = time.time()
        if len(location) > 3 and perfect:
            pre_code, fault_line, changes, post_code = process_file(file, line_number, tokenizer, model, 15,
                                                                    re_rank, top_n_patches)
        else:
            pre_code, fault_line, changes, post_code = process_file(file, line_number, tokenizer, model, beam_width,
                                                                    re_rank, top_n_patches)
        end_time = time.time()
        validator.add_new_patch_generation(pre_code, fault_line, changes, post_code, file, line_number,
                                           end_time - start_time)

        validator.validate()

        subprocess.run('rm -rf ' + '/tmp/' + bug_id, shell=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--bug_id', type=str, default='Chart-1')
    parser.add_argument('--output_folder', type=str, default='result')
    parser.add_argument('--skip_v', action='store_true', default=False)
    parser.add_argument('--re_rank', action='store_true', default=False)
    parser.add_argument('--beam_width', type=int, default=25)
    parser.add_argument('--perfect', action='store_true', default=False)
    parser.add_argument('--top_n_patches', type=int, default=-1)
    args = parser.parse_args()
    print("Run with setting:")
    print(args)
    main([args.bug_id], args.output_folder, args.skip_v, args.beam_width,
         args.re_rank, args.perfect, args.top_n_patches)
