"""Patch for SQLite issues with ChromaDB.

This module monkey patches the sqlite3 module to use the pysqlite3 package,
which provides a newer version of SQLite.
"""

import os
import sys

import pysqlite3

# Save the original sqlite3 module
sys.modules["sqlite3_original"] = sys.modules.get("sqlite3")

# Replace sqlite3 with pysqlite3
sys.modules["sqlite3"] = pysqlite3

# Set environment variable to avoid SQLite version checks in ChromaDB
os.environ["CHROMA_DANGEROUS_DISABLE_SQLITE_VERSION_CHECK"] = "1"

print(f"Patched sqlite3 with pysqlite3 {pysqlite3.sqlite_version}")
