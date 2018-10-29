import setuptools

setuptools.setup(
    name='fusewarp',
    version='0.1.0',
    author='Satoshi Tanaka',
    author_email='stnk20@gmail.com',
    description="Simple support library for image data augmentation",
    url='https://github.com/stnk20/fusewarp',
    packages=setuptools.find_packages(),
    install_requires=["numpy", "scikit-image"],
)
