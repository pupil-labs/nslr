try:
	from .cppnslr import *
except ImportError:
	from .slow_nslr import *
