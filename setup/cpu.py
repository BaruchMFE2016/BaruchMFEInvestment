__author__ = 'Derek Qi'


def cpu_gen(fml):
	def cpu(t, fml=fml):
		fml = fml.replace(' ', '')
		fml = fml.replace('$', 't')
		fml = fml.split('=') # I will assume that there is only one equal sign
		# print(fml)
		t[fml[0]] = eval(fml[1])
		return t
	return cpu
