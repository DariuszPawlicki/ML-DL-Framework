#define PY_SSIZE_T_CLEAN

#include <Python.h>
#include <arrayobject.h>

#include <iostream>
#include <map>


using namespace std;


static PyObject* encode_one_hot(PyObject* self, PyObject* args, PyObject* kwargs) {

	PyArrayObject* labels_list = NULL;
	PyArrayObject* one_hot;

	npy_intp dims[2];

	map<int, int> classes_map;

	int current_label;
	int labels_size;

	int new_numeration = 0;

	void* ptr;

	static char* kwlist[] = { (char*)"labels_list", NULL };
		
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O:encode_one_hot", (char**)kwlist, &labels_list))
		return NULL;
	

	PyArray_Sort(labels_list, 0, NPY_QUICKSORT);

	labels_size = PyArray_SIZE(labels_list);

	for (int i = 0; i < labels_size; i++) {
		ptr = PyArray_GETPTR1(labels_list, i);
		current_label = PyLong_AsLong(PyArray_GETITEM(labels_list, ptr));

		if (classes_map.find(current_label) == classes_map.end()) {
			classes_map[current_label] = new_numeration;
			new_numeration++;
		}
	}

	dims[0] = labels_size;
	dims[1] = (int)classes_map.size();

	one_hot = (PyArrayObject*)PyArray_ZEROS(2, dims, NPY_INT, 0);

	for (int i = 0; i < labels_size; i++) {
		current_label = classes_map[PyLong_AsLong(PyArray_GETITEM(labels_list, PyArray_GETPTR1(labels_list, i)))];
		ptr = PyArray_GETPTR2(one_hot, i, current_label);

		PyArray_SETITEM(one_hot, ptr, PyLong_FromLong(1));
	}
	
	return PyArray_Return(one_hot);
}


static PyMethodDef preprocessing_methods[] = {
	{"encode_one_hot", (PyCFunction)encode_one_hot, METH_VARARGS | METH_KEYWORDS, 
	"encode_one_hot(labels_list: list)\n\n"
	"Returns numpy array of one-hot encoded vectors."},

	{NULL, NULL, 0, NULL}
};

static struct PyModuleDef preprocessing_extension = {
	PyModuleDef_HEAD_INIT,
	"preprocessing_extension",
	"Processing data module",
	-1,
	preprocessing_methods
};

PyMODINIT_FUNC PyInit_preprocessing_extension(void) {
	import_array();

	return PyModule_Create(&preprocessing_extension);
}