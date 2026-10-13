# Source: github.com/wangrp1997/tendon-spin (MIT), exported Boya model contract.
# This creates a local compatibility configuration only; it does not import a
# simulator, reinstall an environment, execute physics or overwrite existing files.
"""Initialize local model configuration from public assets for a fresh checkout."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1],
                        help='Project checkout containing assets/config')
    args = parser.parse_args()
    root = args.root.resolve()
    source = root / 'assets/config/boya_native_contract.json'
    content = source.read_bytes()
    contract = json.loads(content)
    if not all(key in contract for key in ('joints', 'actuators', 'couplings', 'object', 'physics_dt')):
        raise ValueError('Public Boya model contract is incomplete')
    target = root / 'docs/data/boya_native_contract.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        print('Preserved existing local model configuration.')
    else:
        with target.open('xb') as file:
            file.write(content)
        print('Created local model configuration from assets/config.')


if __name__ == '__main__':
    main()
