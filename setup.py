from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.command.develop import develop
import os
import sys
import site

class PostInstallCommand(install):
    def run(self):
        install.run(self)
        self._post_install()

class PostDevelopCommand(develop):
    def run(self):
        develop.run(self)
        self._post_install()

    def _post_install(self):
        # Append the package directory to PYTHONPATH
        package_path = os.path.abspath(os.path.dirname(__file__))
        if package_path not in sys.path:
            sys.path.append(package_path)
        # Add the package path to the user site-packages
        site.addsitedir(package_path)
        
        # OS-independent way to set the PYTHONPATH environment variable
        self._set_pythonpath(package_path)
        
        print(f"Added {package_path} to PYTHONPATH and site-packages")

    def _set_pythonpath(self, package_path):
        # Determine the shell type
        shell = os.getenv('SHELL')
        if shell and 'bash' in shell:
            profile = os.path.expanduser('~/.bashrc')
        elif shell and 'zsh' in shell:
            profile = os.path.expanduser('~/.zshrc')
        elif os.name == 'nt':
            profile = os.path.expanduser('~/_profile')
        else:
            profile = os.path.expanduser('~/.profile')

        # Append the package path to the profile script
        with open(profile, 'a') as file:
            file.write(f'\nexport PYTHONPATH="{package_path}:$PYTHONPATH"\n')

setup(
    name='xami_dataset',
    version='0.1',
    packages=find_packages(),
    install_requires=[
    ],
    include_package_data=True,
    package_data={
        '': ['*.yaml', '*.json'],
    },
    description='The dataset for XAMI (XMM-Newton optical Artefact Mapping for astronomical Instance segmentation)',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Elisabeta-Iulia Dima and ESA contributors',
    author_email='iuliaelisa15@yahoo.com',
    url='https://github.com/ESA-Datalabs/XAMI-dataset',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
    cmdclass={
        'install': PostInstallCommand,
        'develop': PostDevelopCommand,
    },
)
