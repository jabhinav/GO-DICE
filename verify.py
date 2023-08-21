import os
import pickle
import sys

import tensorflow as tf

import logging
from domains.PnPExpert import PnPExpert, PnPExpertTwoObj
from her.rollout import RolloutWorker
from utils.env import get_PnP_env

logger = logging.getLogger(__name__)


def run_verify(args, model_dir):
	args.log_wandb = False
	
	# ############################################# TESTING #################################################### #
	print("\n------------- Verifying SkilledActor at {} -------------".format(model_dir))
	from models.GODICE import Agent as Agent_skilledDemoDICE
	agent = Agent_skilledDemoDICE(args)
	
	logger.info("Loading Model Weights from {}".format(model_dir))
	agent.load_model(dir_param=model_dir)
	
	resume_files = os.listdir('./pnp_data/two_obj_0_1_train_env_states')
	resume_files = [f for f in resume_files if f.endswith('.pkl')]
	resume_states = []
	for f in resume_files:
		with open(os.path.join('./pnp_data/two_obj_0_1_train_env_states', f), 'rb') as file:
			resume_states.append(pickle.load(file))
	
	agent.visualise(
		use_expert_skill=False,
		use_expert_action=False,
		resume_states=None,
		num_episodes=2,
	)

	# # ############################################# EXPERT WORKER ############################################# #
	# exp_env = get_PnP_env(args)
	
	# if args.two_object:
	# 	expert_policy = PnPExpertTwoObj(expert_behaviour=args.expert_behaviour, wrap_skill_id=args.wrap_skill_id)
	# else:
	# 	expert_policy = PnPExpert()
	#
	# policy_worker.policy.use_expert_policy = True
	# policy_worker.policy.use_expert_goal = True
	# policy_worker.policy.use_expert_skill = True
	
	# # ###################################### TEST ON TRAIN ENVS ####################################### #
	# env_state_dir = os.path.join(
	# 	args.dir_data, '{}_env_states_train'.format(
	# 		'two_obj_{}'.format(args.expert_behaviour) if args.two_object else 'single_obj'))
	# env_state_paths = [os.path.join(env_state_dir, 'env_{}.pkl'.format(n)) for n in range(args.train_demos)]
	# test_demos = min(args.test_demos, args.train_demos)
	# for n in range(test_demos):
	# 	print('Train Demo: ', n)
	# 	logger.info("\nTesting Train Demo {}".format(n))
	#
	# 	with open(env_state_paths[n], 'rb') as handle:
	# 		init_state_dict = pickle.load(handle)
	# 		init_state_dict['goal'] = init_state_dict['goal'].numpy() if tf.is_tensor(init_state_dict['goal']) else \
	# 			init_state_dict['goal']
	#
	# 	exp_episode, _ = expert_worker.generate_rollout(init_state_dict=init_state_dict)
	#
	# 	policy_episode, _ = policy_worker.generate_rollout(init_state_dict=init_state_dict, expert_assist=True)
	#
	# ###################################### TEST ON VAL ENVS ####################################### #
	# env_state_dir = os.path.join(
	# 	args.dir_data, '{}_env_states_val'.format(
	# 		'two_obj_{}'.format(args.expert_behaviour) if args.two_object else 'single_obj'))
	# env_state_paths = [os.path.join(env_state_dir, 'env_{}.pkl'.format(n)) for n in range(args.val_demos)]
	#
	# avg_skill_pred_accuracy = 0.
	# avg_skill_term_pred_accuracy = 0.
	# avg_skill_term_pred_precision = 0.
	# avg_skill_term_pred_recall = 0.
	# test_demos = min(args.test_demos, args.val_demos)
	# for n in range(test_demos):
	# 	print('Val Demo: ', n)
	# 	logger.info("\nTesting Val Demo {}".format(n))
	#
	# 	with open(env_state_paths[n], 'rb') as handle:
	# 		init_state_dict = pickle.load(handle)
	# 		init_state_dict['goal'] = init_state_dict['goal'].numpy() if tf.is_tensor(init_state_dict['goal']) else \
	# 			init_state_dict['goal']
	#
	# 	exp_episode, _ = expert_worker.generate_rollout(init_state_dict=init_state_dict)
	#
	# 	policy_episode, _ = policy_worker.generate_rollout(init_state_dict=init_state_dict, expert_assist=True)
	#
	#
	# # Current goal
	# policy_goals = agent_test.model.goal_pred(exp_episode['prev_skills'][0],
	# 										  exp_episode['states'][0, :-1, :],
	# 										  exp_episode['env_goals'][0, :-1, :])
	#
	# # Termination of prev skills
	# expert_prev_skills = tf.argmax(exp_episode['prev_skills'][0], axis=-1)
	# expert_terminate_expert_prev_skills = tf.cast(tf.not_equal(expert_prev_skills, tf.argmax(exp_episode['curr_skills'][0], axis=-1)), dtype=tf.int32)
	# expert_terminate_expert_prev_skills = expert_terminate_expert_prev_skills.numpy().tolist()
	# policy_terminate_expert_prev_skills = []
	# for t, c in enumerate(expert_prev_skills.numpy().tolist()):
	# 	prev_skill_term_net = getattr(agent_test.model, 'skill_term_{}'.format(c))
	# 	terminate_skill_logit = prev_skill_term_net(exp_episode['states'][:, t, :], exp_episode['env_goals'][:, t, :], exp_episode['curr_goals'][:, t, :])
	# 	terminate_skill = tf.cast(tf.nn.sigmoid(terminate_skill_logit) > 0.5, tf.float32)
	# 	policy_terminate_expert_prev_skills.append(terminate_skill[0])
	# policy_terminate_expert_prev_skills = tf.stack(policy_terminate_expert_prev_skills, axis=0)
	# policy_terminate_expert_prev_skills = tf.reshape(policy_terminate_expert_prev_skills, [-1]).numpy().tolist()
	# policy_terminate_expert_prev_skills = [int(t) for t in policy_terminate_expert_prev_skills]
	# logger.info(f'[PrevSkillTerminate with ExpertState, ExpertCurrGoal] Expert: {expert_terminate_expert_prev_skills}')
	# logger.info(f'[PrevSkillTerminate with ExpertState, ExpertCurrGoal] Policy: {policy_terminate_expert_prev_skills}')
	# acc = np.sum(np.array(policy_terminate_expert_prev_skills) == np.array(expert_terminate_expert_prev_skills)) / len(expert_terminate_expert_prev_skills)
	# prec = np.sum(np.array(policy_terminate_expert_prev_skills) * np.array(expert_terminate_expert_prev_skills)) / np.sum(np.array(policy_terminate_expert_prev_skills))
	# recall = np.sum(np.array(policy_terminate_expert_prev_skills) * np.array(expert_terminate_expert_prev_skills)) / np.sum(np.array(expert_terminate_expert_prev_skills))
	# avg_skill_term_pred_accuracy += acc
	# avg_skill_term_pred_precision += prec
	# avg_skill_term_pred_recall += recall
	# logger.info(f'[PrevSkillTerminate with ExpertState, ExpertCurrGoal] Prev Skill Termination Prediction Accuracy: {acc}')
	# logger.info(f'[PrevSkillTerminate with ExpertState, ExpertCurrGoal] Prev Skill Termination Prediction Precision: {prec}')
	# logger.info(f'[PrevSkillTerminate with ExpertState, ExpertCurrGoal] Prev Skill Termination Prediction Recall: {recall}')
	#
	# # Get the policy-predicted curr skill
	# c_pred_with_exert_sg = agent_test.model.skill_pred(exp_episode['states'][0, :-1, :], exp_episode['env_goals'][0, :-1, :], exp_episode['curr_goals'][0])
	# c_pred_with_exert_sg = tf.nn.softmax(c_pred_with_exert_sg, axis=-1)
	# c_pred_with_exert_sg = tf.argmax(c_pred_with_exert_sg, axis=-1).numpy().tolist()
	#
	# c_pred_with_expert_s = agent_test.model.skill_pred(exp_episode['states'][0, :-1, :], exp_episode['env_goals'][0, :-1, :], policy_goals)
	# c_pred_with_expert_s = tf.nn.softmax(c_pred_with_expert_s, axis=-1)
	# c_pred_with_expert_s = tf.argmax(c_pred_with_expert_s, axis=-1).numpy().tolist()
	#
	# exp_skills = tf.argmax(exp_episode['curr_skills'][0], axis=-1).numpy().tolist()
	# logger.info(f'[CurrSkill with ExpertState, ExpertCurrGoal] Expert: {exp_skills}')
	# logger.info(f'[CurrSkill with ExpertState, ExpertCurrGoal] Policy: {c_pred_with_exert_sg}')
	# acc = np.sum(np.array(c_pred_with_exert_sg) == np.array(exp_skills)) / len(exp_skills)
	# avg_skill_pred_accuracy += acc
	# logger.info(f'[CurrSkill with ExpertState, ExpertCurrGoal] Curr Skill Prediction Accuracy: {acc}')
	# logger.info(f'[CurrSkill with ExpertState, PolicyCurrGoal] Policy: {c_pred_with_expert_s}')
	# acc = np.sum(np.array(c_pred_with_expert_s) == np.array(exp_skills)) / len(exp_skills)
	# logger.info(f'[CurrSkill with ExpertState, PolicyCurrGoal] Curr Skill Prediction Accuracy: {acc}')
	#
	# # ------------------------- Policy Rollout with Expert State, Policy Curr Goal --------------------------- #
	#
	# # Get the latent modes with options policy
	# policy_curr_skills_on_its_own = policy_episode['curr_skills'][0]
	# policy_curr_skills_on_its_own = tf.argmax(policy_curr_skills_on_its_own, axis=-1).numpy().tolist()
	# logger.info(f'[CurrSkill with PolicyState, PolicyCurrGoal] Policy: {policy_curr_skills_on_its_own}')
	# # # Monitor the metrics during each episode
	# # delta_AG = np.linalg.norm(policy_episode['quality'].numpy()[0], axis=-1)
	# # fig_path = os.path.join(args.dir_plot, 'Test_{}_{}_wAgentPolicy.png'.format(test_model, n))
	# # plot_metrics(
	# # 	metrics=[delta_AG],
	# # 	labels=['|G_pred - AG_curr|'],
	# # 	fig_path=fig_path, y_label='Metrics', x_label='Steps'
	# # )
# logger.info("\n\nResults:")
# logger.info(f'[PrevSkillTerminate with ExpertState, ExpertCurrGoal] Average Prev Skill Termination Prediction Accuracy: {avg_skill_term_pred_accuracy / args.test_demos}')
# logger.info(f'[PrevSkillTerminate with ExpertState, ExpertCurrGoal] Average Prev Skill Termination Prediction Precision: {avg_skill_term_pred_precision / args.test_demos}')
# logger.info(f'[PrevSkillTerminate with ExpertState, ExpertCurrGoal] Average Prev Skill Termination Prediction Recall: {avg_skill_term_pred_recall / args.test_demos}')
# logger.info(f'[CurrSkill with ExpertState, ExpertCurrGoal] Average Curr Skill Prediction Accuracy: {avg_skill_pred_accuracy / args.test_demos}')
