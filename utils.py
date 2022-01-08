import os

def in_colab():
    if os.getenv('HOME') == '/root':
        return True
    return False