from setuptools import setup

package_name = 'normal_camera'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/grid_2x2_launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='gb',
    maintainer_email='gb@todo.todo',
    description='Multi camera streaming with grid viewer',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'camera_0_publisher = normal_camera.camera_0_publisher:main',
            'camera_2_publisher = normal_camera.camera_2_publisher:main',
            'camera_4_publisher = normal_camera.camera_4_publisher:main',
            'camera_6_publisher = normal_camera.camera_6_publisher:main',
            'grid_viewer = normal_camera.grid_viewer:main',
        ],
    },
)
