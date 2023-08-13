def get_multiple_configs():
	LoC = [
		# {
		# 	# Two Object: GoFar
		# 	"algo": 'GoFar',
		# 	"skill_supervision": 'none',
		# 	"expert_demos": 25,
		# 	"offline_demos": 75,
		# },
		# {
		# 	# Two Object: BC
		# 	"algo": 'BC',
		# 	"skill_supervision": 'none',
		# 	"expert_demos": 25,
		# 	"offline_demos": 75,
		# },
		# {
		# 	# Two Object: DemoDICE
		# 	"algo": 'DemoDICE',
		# 	"skill_supervision": 'none',
		# 	"expert_demos": 25,
		# 	"offline_demos": 75,
		# },
		# {
		# 	# Two Object: Semi 0.25, wrap 1
		# 	"algo": 'SkilledDemoDICE',
		# 	"skill_supervision": 'semi',
		# 	"wrap_level": '1',
		# 	"expert_demos": 25,
		# 	"offline_demos": 75,
		# },
		# {
		# 	# Two Object: Semi 0.25, wrap 2
		# 	"algo": 'SkilledDemoDICE',
		# 	"skill_supervision": 'semi',
		# 	"wrap_level": '2',
		# 	"expert_demos": 25,
		# 	"offline_demos": 75,
		# },
		{
			# Two Object: Semi 0.10, wrap 0
			"algo": 'SkilledDemoDICE',
			"skill_supervision": 'none:0.25',
			"num_skills": 6,
			"expert_demos": 25,
			"offline_demos": 75,
		},
		{
			# Two Object: Semi 0.10, wrap 1
			"algo": 'SkilledDemoDICE',
			"skill_supervision": 'none:0.25',
			"num_skills": 3,
			"expert_demos": 25,
			"offline_demos": 75,
		},
		{
			# Two Object: Semi 0.10, wrap 2
			"algo": 'SkilledDemoDICE',
			"skill_supervision": 'none:0.25',
			"num_skills": 2,
			"expert_demos": 25,
			"offline_demos": 75,
		},
		# {
		# 	# Two Object: Semi 0.10, wrap 1
		# 	"algo": 'SkilledDemoDICE',
		# 	"skill_supervision": 'semi',
		# 	"wrap_level": '1',
		# 	"expert_demos": 10,
		# 	"offline_demos": 90,
		# },
		# {
		# 	# Two Object: Semi 0.10, wrap 2
		# 	"algo": 'SkilledDemoDICE',
		# 	"skill_supervision": 'semi',
		# 	"wrap_level": '2',
		# 	"expert_demos": 10,
		# 	"offline_demos": 90,
		# },
		# {
		# 	# Two Object: Semi 0.50, wrap 0
		# 	"algo": 'SkilledDemoDICE',
		# 	"skill_supervision": 'semi',
		# 	"wrap_level": '0',
		# 	"expert_demos": 50,
		# 	"offline_demos": 50,
		# },
		# {
		# 	# Two Object: Semi 0.50, wrap 1
		# 	"algo": 'SkilledDemoDICE',
		# 	"skill_supervision": 'semi',
		# 	"wrap_level": '1',
		# 	"expert_demos": 50,
		# 	"offline_demos": 50,
		# },
		# {
		# 	# Two Object: Semi 0.50, wrap 2
		# 	"algo": 'SkilledDemoDICE',
		# 	"skill_supervision": 'semi',
		# 	"wrap_level": '2',
		# 	"expert_demos": 50,
		# 	"offline_demos": 50,
		# },
	]
	return LoC
