from setuptools import setup
from distutils.sysconfig import get_python_lib
import os 

setup(name='tnioad',
      version='1.0',
      description='Online Action Detection',
      packages=['tnioad'],
      zip_safe=False)


print("Your Site Packages path:")
lib_path = get_python_lib()
print(lib_path)
for mmpackage in ['mmaction', ]:
    cfg_path = f'configs/{mmpackage}'
    if os.path.islink(cfg_path):
        print(f'unlink {cfg_path}')
        os.unlink(cfg_path)
    mmcfg_path = os.path.join(lib_path, mmpackage, '.mim/configs')
    print(mmcfg_path)
    if os.path.isdir(mmcfg_path):
        print(f"adding {cfg_path}")
        os.symlink(mmcfg_path, cfg_path)

