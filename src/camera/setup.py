from setuptools import setup
import os
from glob import glob

package_name = 'camera'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
  
    data_files=[
    ('share/ament_index/resource_index/packages',
        ['resource/camera']),
    ('share/camera', ['package.xml']),
    (os.path.join('share', 'camera', 'launch'),
        glob('launch/*.py')),
],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='gb',
    maintainer_email='gb@example.com',
    description='Stereo camera stitching package with 4 cameras in 2x2 grid',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'stereo_stitch_0 = camera.stereo_stitch_0:main',
            'stereo_stitch_2 = camera.stereo_stitch_2:main',
            'stereo_stitch_4 = camera.stereo_stitch_4:main',
            'stereo_stitch_6 = camera.stereo_stitch_6:main',
            'combined_view = camera.combined_view_node:main',
        ],
    },
)