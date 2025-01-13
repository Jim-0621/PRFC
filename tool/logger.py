import os

class Logger:

    def __init__(self, logfile):
        self.logfile = logfile
        os.makedirs(os.path.dirname(logfile), exist_ok=True)
        open(logfile, "a").close()

    def log(self, msg, out=False):
        with open(self.logfile, "a+") as logfile:
            logfile.write(msg)
            logfile.write("\n")
        if out:
            print(msg)

    def logo(self, msg):
        self.log(str(msg), True)

