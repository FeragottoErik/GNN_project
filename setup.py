from setuptools import setup, find_packages

setup(
    name='coronary_artery_classification',  # Replace with your desired package name
    version='0.1.0',            # Choose an appropriate version number
    description='Package for coronary artery classification in RCA and LCA based on graph neural networks.',
    author='Feragotto Erik',
    author_email='erik@feragotto.com',
    url='https://github.com/FeragottoErik/GNN_project.git',  # Replace with your repository URL
    packages=find_packages(),
    install_requires=[
        'hcatnetwork>=2023.11.14.post0',
        'cython',
        'numpy',
        'scipy',
        'matplotlib',
        'PyQt6',
        'palettable',
        'NetworkX',
        'PyVis'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
)