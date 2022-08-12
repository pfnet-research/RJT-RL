import os
import re

from setuptools import find_packages, setup

install_requires = []

with open("requirements.txt") as fp:
    install_requires += [
        line for line in fp.readlines() if not re.search("^[-=#].*", line)
    ]

dev_requires = ""

with open("requirements-dev.txt") as fp:
    dev_requires += fp.read()

with open("README.md") as f:
    readme = f.read()

here = os.path.abspath(os.path.dirname(__file__))
# Get __version__ variable
exec(open(os.path.join(here, "rjt_rl", "_version.py")).read())

package_data = {"rjt_rl": ["rl/rewards/fpscores.pkl.gz"]}

setup(
    name="RJT-RL",
    version=__version__,  # NOQA
    description="Reversible Junction Tree Reinforcement Learning",
    long_description=readme,
    author="Ryuichiro Ishitani",
    author_email="ishitani@preferred.jp",
    url="https://github.com/pfnet-research/RJT-RL",
    packages=find_packages(),
    install_requires=install_requires,
    extras_require={"dev": dev_requires},
    include_package_data=True,
    package_data=package_data,
)
