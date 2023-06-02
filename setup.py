from setuptools import setup

setup(
    name='TensorQ',
    version='0.1.0',
    author='Xian-he Zhao',
    author_email='648124022@qq.com',
    packages=['tensorq'],
    url='https://github.com/Xenon-zhao/TensorQ',
    license='LICENSE',
    description='An awesome package that can simulate quantum circuit based on tensor network',
    long_description=open('README.md').read(),
    install_requires=[
        "numpy",
        "artensor"
    ],
)