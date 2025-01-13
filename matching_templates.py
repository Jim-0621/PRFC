import re
import javalang
import javalang.tokenizer


# MT6
def mutate_common_word(code):
    replacements = {
        "max": "min",
        "min": "max",
        "all": "any",
        "any": "all",
        "true": "false",
        "false": "true",
        "int": "double",
        "double": "int",
        "<=": "==",
        ">=": "==",
        "==": "!=",
        "!=": "=="
    }

    replaced_code = []
    for old_str, new_str in replacements.items():
        if old_str in code:
            replaced_code.append((code.replace(old_str, new_str)))

    return replaced_code


# MT7
def match_simple_operator(code):
    ret = [""]
    try:
        tokens = list(javalang.tokenizer.tokenize(code))
    except:
        return []
    current_index = 1
    for token in tokens:
        while current_index < token.position[1]:
            current_index += 1
            ret = [v + " " for v in ret]

        if token.__class__.__name__ == 'Operator':
            for index in range(1, len(ret)):
                ret[index] = ret[index] + token.value
            ret.append(ret[0] + "2mask2")
            ret[0] = ret[0] + token.value

        current_index += len(token.value)
        if not token.__class__.__name__ == 'Operator':
            ret = [v + token.value for v in ret]

    ret = [v.replace(" 2mask2", "<mask>") for v in ret]
    ret = [v.replace("2mask2", "<mask>") for v in ret]
    ret = ret[1:]
    return ret

# MT8.1
def match_conditional_expression1(code):
    ret = []
    bef_code = code.split("(")
    aft_code = code.split(")")
    pre_code = bef_code[0] + "("
    post_code = ")" + aft_code[-1]
    ret.append((pre_code, post_code))

    return ret

# MT8.3-4
def match_conditional_expression2(code):
    ret = []
    if re.match(r".*if\s?\(.+\)\s?{$", code):
        s_code = code.split(")")
        pre_code = ")".join(s_code[:-1])
        post_code = ")" + s_code[-1]
        ret.append((pre_code + ' &&', post_code))
        ret.append((pre_code + ' ||', post_code))

    return ret

# MT9
def match_return_expression(code):
    ret = []
    if re.match(r"return\s?.+\;", code):
        ret.append(('return', ';'))
        s_code = code.split(";")
        pre_code = " ".join(s_code[:-1])
        ret.append((pre_code + " &&", ";"))
        ret.append((pre_code + " ||", ";"))
    return ret

# MT10.1
def match_calling_function(code, tokenizer):
    ret = []
    matches = re.finditer(r"[^)(\s]+\([^)(]+\)", code)
    for match in matches:
        matched_code = match.group()
        sc = code.split(matched_code)
        if len(sc) != 2:
            continue
        matched_code.split("(")
        ret.append((sc[0] + "<mask>" + "(" + "".join(matched_code.split("(")[1:]) + sc[1],
                    tokenizer(matched_code.split("(")[0], return_tensors='pt')['input_ids'].size()[1] - 2))

    return ret

# MT10.
def match_function_api_call(code, tokenizer):
    ret = []
    matches = re.finditer(r"\([^)(]+\)", code)
    for match in matches:
        matched_code = match.group()
        sc = code.split(matched_code)
        new_code = "(<mask>)".join(sc)

        if new_code not in [v[0] for v in ret]:
            ret.append((new_code, tokenizer(matched_code[1:-1], return_tensors='pt')['input_ids'].size()[1] - 2))

    return ret

# MT10.2-3
def _match_function_multi_input_api_call_generate_template(matched_code, tokenizer):
    ret = []
    parameters = matched_code.split(",")
    max = 0
    for parameter in parameters:
        size = tokenizer(parameter, return_tensors='pt')['input_ids'].size()[1] - 2
        if size > max:
            max = size

    ret.append(("(<mask>, " + matched_code + ")", max))
    ret.append(("(" + matched_code + ",<mask>" + ")", max))

    for index, parameter in enumerate(parameters):
        new_code = "("
        for jindex in range(len(parameters)):
            add_code = "<mask>"
            if index != jindex:
                add_code = parameters[jindex]

            if jindex != 0:
                new_code += "," + add_code
            else:
                new_code += add_code
        new_code += ")"
        ret.append((new_code, max))

    return ret

# MT10.2-3
def match_function_multi_input_api_call(code, tokenizer):

    ret = []
    matches = re.finditer(r"\([^)(]+,[^)(]+\)", code)
    for match in matches:
        matched_code = match.group()
        sc = code.split(matched_code)
        if len(sc) != 2:
            continue
        matched_code = matched_code[1:-1]
        for p_code in _match_function_multi_input_api_call_generate_template(matched_code, tokenizer):
            ret.append((sc[0] + p_code[0] + sc[1], p_code[1]))
    return ret



# MT10
def mutate_method_expression(code, tokenizer):
    ret = []
    ret.extend(match_function_api_call(code, tokenizer))
    ret.extend(match_function_multi_input_api_call(code, tokenizer))
    ret.extend(match_calling_function(code, tokenizer))

    return ret