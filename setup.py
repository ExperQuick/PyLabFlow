# Copyright 2024 BBEK Anand
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License. 


from setuptools import setup, find_packages

def get_version():
    version_file = os.path.join('src', 'plf', '_version.py')
    with open(version_file) as f:
        for line in f:
            if line.strip().startswith('__version__'):
                # Extract the version string between quotes
                parts = line.split('=', 1)
                if len(parts) == 2:
                    version_str = parts[1].strip().strip('"\'')
                    return version_str
                break
    raise RuntimeError("Unable to find version string in _version.py.")

setup(
    name='PyLabFlow',
    version=get_version(),
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    include_package_data=True,
    install_requires=[
        'pandas'
    ],
    author='BBEK-Anand',
    author_email='',
    description='PyLabFlow is a lightweight framework that simplifies experiment management, reducing setup time with reusable components for training, logging, and checkpointing. It streamlines workflows, making it ideal for fast and efficient experimentation.',
    long_description=open('./README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/ExperQuick/PyLabFlow',
    license="Apache-2.0",
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)
