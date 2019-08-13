def handle_integers( params ):

	new_params = {}
	for k, v in params.items():
		if type( v ) == float and int( v ) == v:
			new_params[k] = int( v )
		else:
			new_params[k] = v
	
	return new_params
