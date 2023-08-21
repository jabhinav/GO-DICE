def get_multiple_configs():
    LoC = [
        {
            # Three Object: None 0.25, c_dim 3
            "algo": 'GODICE',
            "skill_supervision": 'semi:0.50ext',
            "wrap_level": '0',
            "expert_demos": 50,
            "offline_demos": 50,
            "update_target_interval": 50,
            "actor_polyak": 0.50,
            "replay_regularization": 0.05,
            "max_time_steps": 20000,
        },
        {
            # Three Object: None 0.25, c_dim 3
            "algo": 'GODICE',
            "skill_supervision": 'none:0.50ext',
            "num_skills": 3,
            "expert_demos": 50,
            "offline_demos": 50,
            "update_target_interval": 50,
            "actor_polyak": 0.50,
            "replay_regularization": 0.05,
            "max_time_steps": 20000,
        },



    ]
    return LoC
