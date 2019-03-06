from distutils.core import setup, Extension
import os



setup(name='symlens',
      version='0.1',
      description='Symbolic Mode-Coupling Evaluation Code',
      url='https://github.com/simonsobs/symlens',
      author='Mathew Madhavacheril',
      author_email='mathewsyriac@gmail.com',
      license='BSD-2-Clause',
      packages=['symlens'],
      package_dir={'symlens':'symlens'},
      zip_safe=False)
