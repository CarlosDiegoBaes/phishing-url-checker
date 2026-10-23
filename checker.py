"""Score URLs from the command line.

The implementation lives in the ``phishguard`` package. This file remains so
``python checker.py <url>`` keeps working.
"""

from phishguard.cli import main

if __name__ == "__main__":
    main()
