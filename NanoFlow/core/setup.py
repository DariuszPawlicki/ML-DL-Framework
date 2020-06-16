from setuptools import Extension, setup
import numpy


def main():
    module = Extension("data_processing", sources = ["data_processing.cpp"],
                       include_dirs = [numpy.get_include() + "\\numpy"])

    setup(name = "data_processing",
          version = "0.1a",
          description = "C++ extension for processing large datasets.",
          author_email = "darekpl9@gmail.com",
          install_requires = ["numpy"],
          ext_modules = [module])


if __name__ == '__main__':
    main()