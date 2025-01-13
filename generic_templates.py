import javalang
import javalang.tokenizer


def remove_redudant(ret):
    real_ret = []
    past_code = set()
    for c in ret:
        if "Move Statement Position" not in c[2]:
            patch_code = c[0].replace(" ", "").replace("#", "").replace("\n", "").replace("\t", "")
            if patch_code in past_code:
                continue

            past_code.add(patch_code)

        real_ret.append(c)

    return real_ret


# MT1
def insert_condition_checker(code):
    ret_list = [
        ('if( ', ' ) { ' + code + ' } '),
        ('if( ', ' ) { return; } ' + code),
        ('if( ', ' ) { return true; } ' + code),
        ('if( ', ' ) { return false; } ' + code),
        ('if( ', ' ) { return 0; } ' + code),
        ('if( ', ' ) { return -1; } ' + code),
        ('if( ', ' ) { return null; } ' + code),
        ('if( ', ' ) { continue; } ' + code),
        ('if( ', ' ) { throw new IllegalArgumentException("IllegalArgumentException"); } ' + code),
    ]
    return ret_list


# MT5.2
def parse_code_end(code):
    ret_list = []
    string_builder = ""
    try:
        tokens = list(javalang.tokenizer.tokenize(code))
    except:
        return ret_list
    tokens.reverse()
    current_index = len(code) + 1

    for token in tokens:
        current_index -= len(token.value)
        if string_builder != "":
            ret_list.append(string_builder)

        while current_index > token.position[1]:
            current_index -= 1
            string_builder = " " + string_builder

        string_builder = token.value + string_builder

    return ret_list


# MT5.3
def parse_code_begin(code):
    ret_list = []
    string_builder = ""
    try:
        tokens = list(javalang.tokenizer.tokenize(code))
    except:
        return ret_list
    current_index = 1
    for token in tokens:
        while current_index < token.position[1]:
            current_index += 1
            string_builder += " "

        if string_builder != "":
            ret_list.append(string_builder)

        string_builder += token.value
        current_index += len(token.value)

    return ret_list


# MT5.2-3
def mutate_line_expression1(code):
    ret = []

    for pre_code in parse_code_begin(code):
        ret.append((pre_code, ""))

    for post_code in parse_code_end(code):
        ret.append(("", post_code))

    return ret


# MT5.4
def mutate_line_expression2(code):
    ret = []

    list1 = parse_code_begin(code)
    list2 = parse_code_end(code)
    len_list = len(list1)
    for i in range(0, len_list - 1):
        ret.append((list1[i], list2[len_list - i - 2]))

    return ret
