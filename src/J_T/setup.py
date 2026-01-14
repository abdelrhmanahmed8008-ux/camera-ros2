from setuptools import setup
from glob import glob
import os

package_name = 'J_T'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    entry_points={
        'console_scripts': [
            'combined_view = J_T.combined_view_opt:main',
            'stereo_stitch_node = J_T.stereo_stitch_node_opt:main',
            'calibrate = J_T.calibrate_homography:main',
        ],
    },
)
