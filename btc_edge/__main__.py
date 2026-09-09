"""`python -m btc_edge ...` entry point."""
import sys

from btc_edge.cli import main

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\ninterrupted")
        sys.exit(130)
