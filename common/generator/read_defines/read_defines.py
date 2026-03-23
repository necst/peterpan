import CppHeaderParser
import re
from typing import List

last_loaded_header = None

class LoadedHeader:
    def __init__(self, header_file):
        self.header_file = header_file
        self.header = CppHeaderParser.CppHeader(header_file)
        self.defines = self.header.defines
        self.compiled_defines = {}

    def get_macro_value(self, macro_name):
        if macro_name not in self.compiled_defines:
            self.compiled_defines[macro_name] = re.compile(rf'{macro_name}\s+(.*?)(?://.*|/\*.*|$)')
        regex = self.compiled_defines[macro_name]
        
        for define in self.defines:
            match = regex.match(define)
            if match:
                return match.group(1)
        
        raise ValueError(f'Macro {macro_name} not found in header file {self.header_file}')
        
        return None

def get_macro_value(header_file, macro_name):
    global last_loaded_header
    if last_loaded_header is None or last_loaded_header.header_file != header_file:
        last_loaded_header = LoadedHeader(header_file)
    return last_loaded_header.get_macro_value(macro_name)

def parse_array(array_string, bracket_char='{', separator_char=','):
    closing_bracket_char = '}' if bracket_char == '{' else ']'
    array_string = array_string.strip()
    if array_string[0] == bracket_char and array_string[-1] == closing_bracket_char:
        array_string = array_string[1:-1]
    return [x for x in array_string.split(separator_char)]

# def load_defines(header_file, defines_dict):
#     # structure
#     # defines_dict: name -> type

#     defines_dict = {
#         'DIMENSION': int,
#         'INIT_COLS': list,
#     }

#     for dname, dtype in defines_dict.items():
#         if dtype == list:
#             defines_dict[dname] = parse_array(get_macro_value(header_file, dname))
#         else:
#             defines_dict[dname] = dtpye(get_macro_value(header_file, dname))


if __name__ == '__main__':
    print(int(get_macro_value('constants.h', 'DIMENSION')))
    print(int(get_macro_value('constants.h', 'INTERVAL_NUMBER')))
    print([int(x) for x in parse_array(get_macro_value('constants.h', 'INIT_COLS'))])
    # print(int(get_macro_value('constants.h', 'ueue'))) # should raise ValueError

