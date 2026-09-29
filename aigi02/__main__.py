import sys
from .cli import main
if __name__=='__main__':
    try:raise SystemExit(main())
    except (ValueError,RuntimeError,FileNotFoundError,FileExistsError) as error:
        print(f'AIGI02 error: {error}',file=sys.stderr)
        raise SystemExit(2)
