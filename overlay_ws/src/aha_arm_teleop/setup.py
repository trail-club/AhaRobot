from setuptools import find_packages, setup

package_name = "aha_arm_teleop"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="trail-club",
    maintainer_email="dev@trail-club.local",
    description="Keyboard teleop for both Astra arms via arm_node joint_command.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "keyboard_teleop = aha_arm_teleop.keyboard_teleop_node:main",
        ],
    },
)
