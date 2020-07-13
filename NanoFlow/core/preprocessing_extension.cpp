#define PY_SSIZE_T_CLEAN

#include <Python.h>
#include <arrayobject.h>

#include <iostream>
#include <map>


using namespace std;


static PyObject* encode_one_hot(PyObject* self, PyObject* args, PyObject* kwargs) {

	PyArrayObject* labels_list;
	PyArrayObject* one_hot;

	npy_intp dims[2];

	int current_label;
	int labels_size;

	void* ptr;

	static char* kwlist[] = { (char*) "labels_list", NULL };
		
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O:encode_one_hot", (char**) kwlist, &labels_list))
		return NULL;
		

	labels_size = PyArray_SIZE(labels_list);

	dims[0] = labels_size;
	dims[1] = PyLong_AsLong(PyArray_Max(labels_list, 0, NULL)) + 1; // If max. class number is e.g. 5
																   //  then class total classes count is 6:
																  //   0,1,2,3,4,5

	one_hot = (PyArrayObject*) PyArray_Zeros(2, dims, PyArray_DescrFromType(NPY_INT), 0);

	for (int i = 0; i < labels_size; i++) {
		current_label = PyLong_AsLong(PyArray_GETITEM(labels_list, PyArray_GETPTR1(labels_list, i)));
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