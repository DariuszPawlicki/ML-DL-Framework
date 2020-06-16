#define NULL 0

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

	~OneHotEncoder() {
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
