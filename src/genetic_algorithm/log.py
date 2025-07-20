import logging
import sys
import inspect
import os
import psutil
from pympler import asizeof

# ─── Logging Setup ────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler("mem_deep.log")
file_handler.setLevel(logging.INFO)

formatter = logging.Formatter(
	"%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def log_deep_var_sizes_safe(namespace, header=None, sort_desc=True):
	"""
	Logs deep sizes for each name→obj in namespace.
	Falls back to shallow size on any exception.
	"""
	entries = []
	for name, val in namespace.items():
		if name.startswith("_"):
			continue
		# try deep size
		try:
			size_bytes = asizeof.asizeof(val)
			method = "deep"
		except Exception:
			size_bytes = sys.getsizeof(val)
			method = "shallow"
		entries.append((name, type(val).__name__, size_bytes, method))

	# sort by size
	entries.sort(key=lambda e: e[2], reverse=sort_desc)

	if header:
		logger.info(header)

	for name, typ, sz, method in entries:
		kb = sz / 1024
		logger.info(f"{name:>20s} {typ:<12s} {kb:8.2f} KiB ({method})")


def report_deep_globals():
	log_deep_var_sizes_safe(globals(), header="=== GLOBAL VARIABLES (deep sizes) ===")


def report_deep_locals():
	# 1) Log current open file descriptors
	proc = psutil.Process(os.getpid())
	try:
		n_fds = proc.num_fds()
		logger.info(f"open_fds: {n_fds}")
	except AttributeError:
		# Windows fallback (handle count)
		n_h = proc.num_handles()
		logger.info(f"open_handles: {n_h}")

	# 2) Log deep sizes of local variables
	caller_locals = inspect.currentframe().f_back.f_locals
	log_deep_var_sizes_safe(caller_locals, header="=== LOCAL VARIABLES (deep sizes) ===")

