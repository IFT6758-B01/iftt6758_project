#!/usr/bin/env python3

class StringColor:
    def __init__(self):
      self.BOLD = '\033[1m'
      self.RED = '\033[31m'
      self.YELLOW = '\033[33m'
      self.GREEN = '\033[32m'
      self.NORMAL = '\033[0m'
      self.CYAN = '\033[36m'
      self.GREY = '\033[37m'
      self.RED_BOLD = self.BOLD + self.RED
      self.YELLOW_BOLD = self.BOLD + self.YELLOW
      self.GREEN_BOLD = self.BOLD + self.GREEN

    def bold(self, text):
        return self.BOLD + text + self.NORMAL

    def error(self, text):
        return self.RED_BOLD + text + self.NORMAL

    def warning(self, text):
        return self.YELLOW_BOLD + text + self.NORMAL

    def success(self, text):
        return self.GREEN_BOLD + text + self.NORMAL

    def info(self, text):
        return self.CYAN + text + self.NORMAL

    def debug(self, text):
        return self.GREY + text + self.NORMAL
