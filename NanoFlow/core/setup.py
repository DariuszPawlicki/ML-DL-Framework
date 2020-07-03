from setuptools import Extension, setup
import numpy


def main():
    module = Extension("preprocessing_extension", sources = ["preprocessing_extension.cpp"],
                       include_dirs = [numpy.get_include() + "/numpy"])

    setup(name = "preprocessing_extension",
          version = "0.1_a",
          description = "C++ extension for processing large datasets.",
          author_email = "darekpl9@gmail.com",
          install_requires = ["numpy"],
          ext_modules = [module])


if __name__ == '__main__':
    main()