from setuptools import setup

def readme():
    with open('README.md') as f:
        README = f.read()
    return README

def parse_requirements(filename):
    with open(filename, "r") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="papyrus-matching",
    version="0.2.0",
    description="A Python package for matching papyrus fragments with Deep Learning.",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/mesnico/papyrus-matching",
    author="Nicola Messina, Fabio Carrara",
    author_email="nicola.messina@isti.cnr.it, fabio.carrara@isti.cnr.it",
    license="MIT",
    packages=["papyrus_matching"],
    install_requires=parse_requirements("requirements.txt"),
    entry_points={
        'console_scripts': [
            # command_name = package.module:function
            'compute = papyrus_matching.precompute:main',
            'postprocess = papyrus_matching.postprocess:main',
            'crop_fragments = papyrus_matching.data.crop_fragments:main',
        ],
    },
)