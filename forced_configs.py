def get_multiple_configs():
	LoC = [
		{
			# Two Object: GoFar
			"algo": 'GoFar',
			"skill_supervision": 'none',
		},
		{
			# Two Object: BC
			"algo": 'BC',
			"skill_supervision": 'none',
			"BC_beta": 0.0,
		},
		{
			# Two Object: BC
			"algo": 'BC',
			"skill_supervision": 'none',
			"BC_beta": 1.0,
		},
		{
			# Two Object: DemoDICE
			"algo": 'DemoDICE',
			"skill_supervision": 'none',
		},
		{
			# Two Object: Semi 0.25, wrap 2
			"algo": 'GODICE',
			"skill_supervision": 'semi',
			"wrap_level": '0',
		},
		{
			# Two Object: Semi 0.10, wrap 0
			"algo": 'GODICE',
			"skill_supervision": 'none',
			"num_skills": 2,
		},
	]
	return LoC
