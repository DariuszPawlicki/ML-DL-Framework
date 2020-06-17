#include <Python.h>
#include <arrayobject.h>

#include <iostream>
#include <map>
//#include "OneHotEncoder.cpp"


using namespace std;


static PyObject* encode_one_hot(PyObject* self, PyObject* args) {

	PyArrayObject* labels;
	PyArrayObject* one_hot;

	npy_intp dims[2];

	map<int, int> classes_map;

	int current_label;
	int labels_size;

	int new_numeration = 0;

	void* ptr;

	if (!PyArg_ParseTuple(args, "O", &labels))
		return NULL;

	PyArray_Sort(labels, 0, NPY_QUICKSORT);

	labels_size = PyArray_SIZE(labels);

	for (int i = 0; i < labels_size; i++) {
		ptr = PyArray_GETPTR1(labels, i);
		current_label = PyLong_AsLong(PyArray_GETITEM(labels, ptr));

		if (classes_map.find(current_label) == classes_map.end()) {
			classes_map[current_label] = new_numeration;
			new_numeration++;
		}
	}

	dims[0] = labels_size;
	dims[1] = (int)classes_map.size();

	one_hot = (PyArrayObject*)PyArray_ZEROS(2, dims, NPY_INT, 0);

	for (int i = 0; i < labels_size; i++) {
		current_label = classes_map[PyLong_AsLong(PyArray_GETITEM(labels, PyArray_GETPTR1(labels, i)))];
		ptr = PyArray_GETPTR2(one_hot, i, current_label);

		PyArray_SETITEM(one_hot, ptr, PyLong_FromLong(1));
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