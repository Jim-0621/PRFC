import os
import re
import signal
import subprocess
import time
import javalang

class GVpatches(object):

    def __init__(self, bug_id, testmethods, logger, patch_pool_folder="patches-pool", skip_validation=False):
        self.bug_id = bug_id
        self.pre_codes = []
        self.fault_lines = []
        self.l_changes = []
        self.post_codes = []
        self.files = []
        self.line_numbers = []
        self.num_of_patches = 1
        self.testmethods = testmethods
        self.logger = logger
        self.patch_pool_folder = patch_pool_folder
        self.skip_validation = skip_validation
        self.generation_time = 0
        self.countPS = 0

    def add_new_patch_generation(self, pre_code, fault_line, changes, post_code, file, line_number, n_time):
        self.pre_codes.append(pre_code)
        self.fault_lines.append(fault_line)
        self.l_changes.append(changes)
        self.post_codes.append(post_code)
        self.files.append(file)
        self.line_numbers.append(line_number)
        self.generation_time += n_time

    def checkout_d4j_project(self):
        subprocess.run('rm -rf ' + '/tmp/' + self.bug_id, shell=True)
        subprocess.run("defects4j checkout -p %s -v %s -w %s" % (self.bug_id.split('-')[0],
                                                                 self.bug_id.split('-')[1] + 'b',
                                                                 ('/tmp/' + self.bug_id)),
                       shell=True)


    def write_changes_to_file(self, change, index, masked_line):
        with open(self.files[index], 'w', encoding='utf-8') as f:
            if "Move Statement Position" not in masked_line:
                for line in self.pre_codes[index]:
                    f.write(line)
                f.write(change + '\n')
                for line in self.post_codes[index]:
                    f.write(line)
            else:
                position_number = int(masked_line.split()[-1])
                if position_number < 6:
                    target_index = len(self.pre_codes[index]) - (6 - position_number)
                    for i, line in enumerate(self.pre_codes[index]):
                        if i == target_index:
                            f.write(change + '\n')
                        f.write(line)
                    for line in self.post_codes[index]:
                        f.write(line)
                else:
                    for line in self.pre_codes[index]:
                        f.write(line)
                    target_index = position_number - 5
                    for i, line in enumerate(self.post_codes[index]):
                        if i == target_index:
                            f.write(change + '\n')
                        f.write(line)

    def run_d4j_test(self, prob, index):

        bugg = False
        compile_fail = False
        timed_out = False
        entire_bugg = False
        error_string = ""

        with open(self.files[index], 'r', encoding='utf-8') as f:
            tmpcode = f.read()
            try:
                tokens = javalang.tokenizer.tokenize(tmpcode)
                parser = javalang.parser.Parser(tokens)
                parser.parse()
            except:
                self.logger.logo(prob)
                self.logger.logo("Syntax Error")
                return

        for t in self.testmethods:
            cmd = 'defects4j test -w %s/ -t %s' % (('/tmp/' + self.bug_id), t.strip())
            Returncode = ""
            error_file = open("stderr.txt", "wb")
            child = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=error_file, bufsize=-1,
                                     start_new_session=True)
            while_begin = time.time()
            while True:
                Flag = child.poll()
                if Flag == 0:
                    Returncode = child.stdout.readlines()
                    print(b"".join(Returncode).decode('utf-8'))
                    error_file.close()
                    break
                elif Flag != 0 and Flag is not None:
                    compile_fail = True
                    error_file.close()
                    with open("stderr.txt", "rb") as f:
                        r = f.readlines()
                    for line in r:
                        if re.search(':\serror:\s', line.decode('utf-8')):
                            error_string = line.decode('utf-8')
                            break
                    print(error_string)
                    break
                elif time.time() - while_begin > 15:
                    error_file.close()
                    print('ppp')
                    os.killpg(os.getpgid(child.pid), signal.SIGTERM)
                    timed_out = True
                    break
                else:
                    time.sleep(1)
            log = Returncode
            if len(log) > 0 and log[-1].decode('utf-8') == "Failing tests: 0\n":
                continue
            else:
                print("Failure Output")
                bugg = True
                break
        if not bugg:
            print('So you pass the basic tests, Check if it passes all the test, include the previously passing tests')
            cmd = 'defects4j test -w %s/' % ('/tmp/' + self.bug_id)
            Returncode = ""
            child = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=-1,
                                     start_new_session=True)
            while_begin = time.time()
            while True:
                Flag = child.poll()
                if Flag == 0:
                    Returncode = child.stdout.readlines()
                    break
                elif Flag != 0 and Flag is not None:
                    bugg = True
                    break
                elif time.time() - while_begin > 180:
                    os.killpg(os.getpgid(child.pid), signal.SIGTERM)
                    bugg = True
                    break
                else:
                    time.sleep(1)
            log = Returncode
            print(log)
            if len(log) > 0 and log[-1].decode('utf-8') == "Failing tests: 0\n":
                print('success')
                endtime = time.time()
            else:
                entire_bugg = True

        self.logger.logo(prob)
        if compile_fail:
            self.logger.logo("Compiled Error:")
            self.logger.logo(error_string)
            self.logger.logo("Compilation Failed")
        elif timed_out:
            self.logger.logo("Timed Out")
        elif bugg:
            self.logger.logo("Failed Testcase")
        elif entire_bugg:
            self.logger.logo("Failed Original Testcase")
        else:
            self.logger.logo("Success (Plausible Patch)")
            self.countPS += 1

    def validate(self):
        start_time = time.time()
        print("----------------- BEGIN VALIDATION OF PATCHES -----------------")
        for index in range(len(self.l_changes)):
            self.logger.logo("Validating Patches for file: {} {}".format(self.files[index], self.line_numbers[index]))
            if not self.skip_validation:
                self.checkout_d4j_project()
            for change, prob, masked_line in self.l_changes[index]:
                if time.time() - start_time > 60 * 60 * 4:
                    break
                if not self.skip_validation:
                    self.write_changes_to_file(change, index, masked_line)
                self.logger.logo("----------------------------------------")
                self.logger.logo("Patch Number :{}".format(self.num_of_patches))
                self.logger.logo('- ' + self.fault_lines[index].strip())
                self.logger.logo('+ ' + change.strip())
                self.logger.logo('(Mask Templates)' + masked_line.strip())
                if not self.skip_validation:
                    self.run_d4j_test(prob, index)
                else:
                    self.logger.logo(prob)
                self.num_of_patches += 1
        end_time = time.time()
        self.logger.logo("----------------- FINISH VALIDATION OF PATCHES -----------------")
        self.logger.logo("Patch Generation Time: {}".format(self.generation_time))
        self.logger.logo("Patch Validation Time: {}".format(end_time - start_time))
        self.logger.logo("Success (Plausible Patch): {}".format(self.countPS))