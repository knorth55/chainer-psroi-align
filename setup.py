from setuptools import find_packages
from setuptools import setup


version = '0.0.0'


setup(
    name='psroi_align',
    version=version,
    packages=find_packages(),
    install_requires=open('requirements.txt').readlines(),
    description='',
    long_description=open('README.md').read(),
    author='Shingo Kitagawa',
    author_email='shingogo.5511@gmail.com',
    url='https://github.com/knorth55/chainer-psroi-align',
    license='MIT',
    keywords='machine-learning',
)
