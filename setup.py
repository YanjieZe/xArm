from setuptools import setup, find_packages

setup(
    name='xArm',
    version='0.1.0',
    packages=find_packages(),
    include_package_data=True,
    license=open('LICENSE').read(),
    zip_safe=False,
    description="xArm robotic environment for reinforcement learning",
    author='Yanjie Ze',
    author_email='zeyanjie@sjtu.edu.cn',
    url='https://peract.github.io/',
    keywords=['Robotics', 'Reinforcement Learning', 'Computer Vision'],
)