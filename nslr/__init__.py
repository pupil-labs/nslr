try:
	from .cppnslr import *
except ImportError:
    print("Failed to import C++ NSLR.")
    raise
