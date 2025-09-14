from setuptools import setup, find_packages

setup(
    name='voltllmclient',
    version='0.1.2',
    description='A lightweight client for local LLM APIs like Ollama or OpenWebUI',
    author='Voltur',
    url='https://github.com/stuarttempleton/volt-llm-client',
    license='MIT',
    packages=find_packages(),
    python_requires='>=3.7',
    install_requires=[
        'requests',
        'volt-logger>=0.1.0'  # <-- Add this line
    ],
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Utilities',
    ],
)