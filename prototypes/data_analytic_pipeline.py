class image_classification_pipeline():
	def __init__(self, image_list=None, y=None, fe=None, dr=None, la=None, **kwargs):
		self.feature_extraction = fe
		self.dimensionality_reduction = dr
		self.learning_algorithm = la
		self.image_list = image_list
		self.response = y
		for key, value in kwargs.items():
			self.__setattr__(key, value)

	def run_pipeline(self):
		
		if self.feature_extraction == 'haralick':
