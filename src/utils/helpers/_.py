# Hack for relatives imports to work
import sys, os
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{dir_path}/../../')