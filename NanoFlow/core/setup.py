from distutils.core import Extension, setup


def main():
    module = Extension("dataProcessing", sources = ["data_processing.cpp"])

    setup(name = "dataProcessing",
          version = "1.0",
          ext_modules = [module])

if __name__ == '__main__':
    main()