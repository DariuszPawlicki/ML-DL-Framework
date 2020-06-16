#include <Python.h>
#include <arrayobject.h>

#include <iostream>
//#include "OneHotEncoder.cpp"


using namespace std;


static PyObject* encode_one_hot(PyObject* self, PyObject* args) {

	PyArrayObject* labels;

	int labels_size, classes_num;

	if (!PyArg_ParseTuple(args, "Oii", &labels, &labels_size, &classes_num))
		return NULL;

	npy_intp dims[2] = {labels_size, classes_num};

	PyArrayObject* one_hot = (PyArrayObject*)PyArray_ZEROS(2, dims, NPY_INT, 0);

	int cur_label;
	void* correct_label_pointer;

		for (int i = 0; i < labels_size; i++) {
			cur_label = PyLong_AsLong(PyArray_GETITEM(labels,
										PyArray_GETPTR1(labels, i)));

			correct_label_pointer = PyArray_GETPTR2(one_hot, i, cur_label);

			PyArray_SETITEM(one_hot, correct_label_pointer, PyLong_FromLong(1));
		}

	
	return PyArray_Return(one_hot);
}


static PyMethodDef data_methods[] = {
	{"encode_one_hot", encode_one_hot, METH_VARARGS, 
	"Creates list of one-hot encoded vectors."},

	{NULL, NULL, 0, NULL}
};

static struct PyModuleDef data_processing = {
	PyModuleDef_HEAD_INIT,
	"data_processing",
	"Processing data module",
	-1,
	data_methods
};

PyMODINIT_FUNC PyInit_data_processing(void) {
	import_array();

	return PyModule_Create(&data_processing);
}