import os
import sys
script_path = os.path.abspath(__file__) # i.e. /path/to/dir/foobar.py
script_dir = os.path.dirname(script_path) #i.e. /path/to/dir/
sys.path.append(script_dir) # add submodules to path so that CNNFlattener can import ChannelAttention from within CNNFlattener