from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="data-tool-climatique",
    version="0.1.0",
    author="Votre Nom",
    author_email="votre.email@example.com",
    description="Outil d'analyse du risque climatique pour le hackathon",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/votre-utilisateur/data-tool-climatique",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=[
        'pandas>=1.3.0',
        'numpy>=1.20.0',
        'scikit-learn>=1.0.0',
        'streamlit>=1.10.0',
        'matplotlib>=3.4.0',
        'seaborn>=0.11.0',
        'pydeck>=0.8.0',
        'altair>=4.2.0',
    ],
    entry_points={
        'console_scripts': [
            'climatique=modules.clim_app:main',
        ],
    },
)
