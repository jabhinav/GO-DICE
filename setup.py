from distutils.core import setup

# Python packaging done using instructions from
# https://the-hitchhikers-guide-to-packaging.readthedocs.io/en/latest/index.html
setup(
    name='MDP-Factorial-AMM',
    version='0.1.0',
    author='multiple (see github)',
    author_email='vaibhav.unhelkar@rice.edu',
    packages=[
        'np_mdp',
    ],
    scripts=[],
    url='https://github.com/unhelkarlab/factorial-amm/tree/siAMM_v1/np_mdp',
    license='TBD',
    description='np-MDP repo for Factorial AMM.',
    long_description=open('README.md').read(),
    install_requires=[
        'absl-py>=0.12.0',
        'numpy>=1.16.6',
        'scipy>=1.2.2',
        'tqdm>=4.59.0',
    ],
    extras_require={},
)
