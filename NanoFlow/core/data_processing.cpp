#include <Python.h>
#include <iostream>


using namespace std;


class OneHotEncoder {

	private:

	int* labels;
	int labels_size, classes_num;
	int** one_hot;

	public:

	OneHotEncoder(int* labels, int labels_size, int classes_num) {
		this->labels = labels;
		this->labels_size = labels_size;
		this->classes_num = classes_num;
		this->one_hot = NULL;
	}

	OneHotEncoder() {
		this->labels = NULL;
		this->labels_size = NULL;
		this->classes_num = NULL;
		this->one_hot = NULL;
	}

	~OneHotEncoder(){
		for (int i = 0; i < this->labels_size; i++) {
			delete[] this->one_hot[i];
		}

		delete[] one_hot;
	}

	int** encode_one_hot() {
		
		this->one_hot = new int* [this->labels_size];


		for (int i = 0; i < this->labels_size; i++) {

			this->one_hot[i] = new int[this->classes_num];

			for (int j = 0; j < this->classes_num; j++) {
				this->one_hot[i][j] = 0;
			}

			this->one_hot[i][this->labels[i]] = 1;
		}

		return this->one_hot;
	}
};


static PyObject* encode_one_hot(PyObject* self, PyObject* args) {
		
	int* labels = NULL;
	int labels_size, classes_num = NULL;
	int** one_hot = NULL;

	if (!PyArg_ParseTuple(args, "Oii", &labels, &labels_size, &classes_num))
		return NULL;

	OneHotEncoder encoder(labels, labels_size, classes_num);

	one_hot = encoder.encode_one_hot();

	return Py_BuildValue("O", one_hot);
}

static PyMethodDef dataMethods[] = {
	{"encode_one_hot", encode_one_hot, METH_VARARGS, "Creates list of one-hot encoded vectors."},
	{NULL, NULL, 0, NULL}
};

static struct PyModuleDef dataProcessing = {
	PyModuleDef_HEAD_INIT,
	"dataProcessing",
	"Processing data module",
	-1,
	dataMethods
};

PyMODINIT_FUNC PyInit_dataProcessing(void) {
	return PyModule_Create(&dataProcessing);
}