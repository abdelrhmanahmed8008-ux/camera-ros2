from setuptools import find_packages, setup

package_name = 'cv-AGX-5G'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='gb',
    maintainer_email='gb@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'stereo_stitch_1 = camera.jetson_stereo_stitch:main',
            'stereo_stitch_2 = camera.stereo_stitch_jetson2:main',
            'stereo_stitch_3 = camera.stereo_stitch_jetson3:main',
            'stereo_stitch_4 = camera.stereo_stitch_jetson4:main',
            'combined_view = camera.combined_view:main',
            'fiveG_network_bridge = camera.fiveG_network_bridge:main',
        ],
    },
)
