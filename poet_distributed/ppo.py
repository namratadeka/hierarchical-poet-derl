import os
import logging
import numpy as np
from os.path import join
from .logger import CSVLogger

import torch
from torch.distributions import MultivariateNormal

from .niches.box2d.actor_critic import Actor, Critic

logger = logging.getLogger(__name__)

def initialize_weights(m):
  if isinstance(m, torch.nn.Conv2d):
      torch.nn.init.kaiming_uniform_(m.weight.data,nonlinearity='relu')
      if m.bias is not None:
          torch.nn.init.constant_(m.bias.data, 0)
  elif isinstance(m, torch.nn.BatchNorm2d):
      torch.nn.init.constant_(m.weight.data, 1)
      torch.nn.init.constant_(m.bias.data, 0)
  elif isinstance(m, torch.nn.Linear):
      torch.nn.init.kaiming_uniform_(m.weight.data)
      torch.nn.init.constant_(m.bias.data, 0)

def learn_util(agent, iteration):
    agent.learn(iteration=iteration)

def eval_util(agent1, agent2):
    agent1.set_morph_params(agent2.morph_params)
    score = agent1.eval_agent(agent2.actor)
    return score

class PPO:
    def __init__(self, make_niche, 
                 morph_params, optim_id, 
                 created_at=0, 
                 is_candidate=False,
                 log_file='',
                 model_dir='',
                 parent=-1):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._init_hyperparameters()

        self._build_network()
        self._build_optimizer()

        self.niche = make_niche()
        self.set_morph_params(morph_params)
        self.act_dim = self.niche.model.env.action_space.shape[0]
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var).to(self.device)

        self.optim_id = optim_id
        self.parent = parent

        self.created_at = created_at
        self.start_score = None
        self.score = None

        if not is_candidate:
            self.id = '{}_{}'.format(self.optim_id, self.morph_id)
            log_path = log_file + '/' + log_file.split('/')[-1] + '.' + self.id + '.log'
            self.data_logger = CSVLogger(log_path, [
                'iteration',
                'env',
                'score',
                'parent'
            ])
            logger.info('Optimizer {} created!'.format(optim_id))
            self.model_dir = join(model_dir, self.id)
            os.makedirs(self.model_dir, exist_ok=True)
        
        self.iteration = None
        self.log_data = {}

    def _init_hyperparameters(self):
        self.timesteps_per_batch = 2000
        self.max_timesteps_per_episode = 1000
        self.gamma = 0.9
        self.epochs = 4
        self.clip = 0.2
        self.entropy_beta = 0.1
        self.minibatch_size = 256

    def set_morph_params(self, morph_params):
        self.morph_params = morph_params
        self.niche.model.env.set_morphology(morph_params)
        self.morph_id = 'm_%s'%'_'.join([str(x) for x in morph_params])

    def save_to_logger(self):
        self.log_data['iteration'] = self.iteration
        self.log_data['env'] = self.optim_id
        self.log_data['score'] = self.score
        self.log_data['parent'] = self.parent
        self.data_logger.log(**self.log_data)
        
    def _build_network(self):
        self.actor = Actor().to(self.device)
        self.critic = Critic().to(self.device)

        self.actor.apply(initialize_weights)
        self.critic.apply(initialize_weights)

    def _build_optimizer(self):
        actor_params = list(self.actor.parameters())
        self.actor_optim = torch.optim.Adam(actor_params, lr=0.00025)

        critic_params = list(self.critic.parameters())
        self.critic_optim = torch.optim.Adam(critic_params, lr=0.00025)
    
    def get_action(self, state, actor:Actor):
        mean = actor(state).squeeze()
        dist = MultivariateNormal(mean, self.cov_mat)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.detach().cpu().numpy(), log_prob.detach()

    def compute_rtgs(self, batch_rews):
        batch_rtgs = []
        for ep_rews in reversed(batch_rews):
            discounted_reward = 0
            for rew in reversed(ep_rews):
                discounted_reward = rew + self.gamma*discounted_reward
                batch_rtgs.insert(0, discounted_reward)

        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float).to(self.device)
        return batch_rtgs

    def rollout(self, render=False):
        batch_states = []
        batch_actions = []
        batch_logprobs = []
        batch_rews = []
        batch_lens = []

        t = 0
        while t < self.timesteps_per_batch:
            ep_rews = []
            state = self.niche.model.env.reset()
            done = False

            for ep_t in range(self.max_timesteps_per_episode):
                if render:
                    self.niche.model.env.render()
                t += 1
                batch_states.append(state)
                action, log_prob = self.get_action(state, self.actor)
                for k in range(4):
                    state, rew, done, _ = self.niche.model.env.step(action)
                
                ep_rews.append(rew)
                batch_actions.append(action)
                batch_logprobs.append(log_prob)

                if done:
                    break
            
            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)

        batch_states = torch.tensor(np.array(batch_states), dtype=torch.float).to(self.device)
        batch_actions = torch.tensor(np.array(batch_actions), dtype=torch.float).to(self.device)
        batch_logprobs = torch.tensor(np.array(batch_logprobs), dtype=torch.float).to(self.device)

        batch_rtgs = self.compute_rtgs(batch_rews)

        return batch_states, batch_actions, batch_logprobs, batch_rtgs, batch_lens

    def eval_agent(self, actor:Actor=None, num_episodes:int=5, max_episode_length:int=2000):
        '''
        Evaluate agent over multiple episodes and return average total reward.
        '''
        if actor is None:
            actor = self.actor
        actor.eval()
        reward_list = []
        for i in range(num_episodes):
            total_reward = 0
            state = self.niche.model.env.reset()
            t = 0
            while t < max_episode_length:
                t += 1
                action, _ = self.get_action(state, actor)
                state, reward, done, _ = self.niche.model.env.step(action)
                total_reward += reward
                if done:
                    break
            reward_list.append(total_reward)
        return np.mean(reward_list)

    def update_score(self):
        self.score = self.eval_agent()

    def evaluate_critic(self, states, actions):
        V = self.critic(states).squeeze()

        mean = self.actor(states).squeeze()
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(actions)

        return V, log_probs

    def learn(self, total_timesteps=100000, render=False, iteration=0):
        self.actor.train()
        self.critic.train()
        self.iteration = iteration
        t_so_far = 0
        itr = 0
        while t_so_far < total_timesteps:
            batch_states, batch_actions, batch_logprobs, batch_rtgs, batch_lens = self.rollout(render)

            avg_rew = self.avg_reward_per_episode(batch_rtgs, batch_lens)
            print("[{}] Average episodic reward: {}".format(itr, avg_rew))

            t_so_far += np.sum(batch_lens)

            batch_states, batch_actions, batch_logprobs, batch_rtgs = self.shuffle(batch_states, batch_actions, 
                batch_logprobs, batch_rtgs)

            for i in range(self.epochs):
                for k in range(0, batch_rtgs.shape[0], self.minibatch_size):
                    
                    states = batch_states[k : k + self.minibatch_size]
                    actions = batch_actions[k : k + self.minibatch_size]
                    rtgs = batch_rtgs[k : k + self.minibatch_size]
                    log_probs = batch_logprobs[k : k + self.minibatch_size]

                    V, curr_logprobs = self.evaluate_critic(states, actions)

                    A_k = rtgs - V.clone().detach()
                    #normalize advantages
                    A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

                    ratios = torch.exp(curr_logprobs - log_probs)

                    #surrogate losses
                    surr_1 = ratios * A_k
                    surr_2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

                    entropy = self.entropy_beta * (-(torch.exp(curr_logprobs) * curr_logprobs)).mean()

                    actor_loss = (-torch.min(surr_1, surr_2)).mean() - entropy
                    critic_loss = torch.nn.MSELoss()(V, rtgs)

                    self.actor_optim.zero_grad()
                    actor_loss.backward(retain_graph=True)
                    self.actor_optim.step()

                    self.critic_optim.zero_grad()
                    critic_loss.backward()
                    self.critic_optim.step()
            
            itr += 1
        self.update_score()
        self.save()
        
    def avg_reward_per_episode(self, batch_rtgs, batch_lens):
        episodic_rewards = []
        for i, ep_len in enumerate(batch_lens):
            total_time_so_far = int(np.sum(batch_lens[:i]))
            episodic_rewards.append(batch_rtgs[total_time_so_far:total_time_so_far+ep_len].sum().item())
        
        return np.mean(episodic_rewards)

    def shuffle(self, batch_states, batch_actions, batch_logprobs, batch_rtgs):
        idx = np.random.randint(0, batch_rtgs.shape[0], batch_rtgs.shape[0])
        batch_states = batch_states[idx]
        batch_actions = batch_actions[idx]
        batch_logprobs = batch_logprobs[idx]
        batch_rtgs = batch_rtgs[idx]

        return batch_states, batch_actions, batch_logprobs, batch_rtgs

    def save(self):
        self.actor.eval()
        self.critic.eval()
        save_dict = {
            'iteration': self.iteration,
            'actor_opt': self.actor_opt.state_dict(),
            'critic_opt': self.critic_opt.state_dict(),
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'env': self.optim_id,
            'morph_params': self.morph_params,
            'score': self.score
        }
        model_path = join(self.model_dir, '{}_{}.pth'.format(self.iteration, self.score))
        torch.save(save_dict, model_path)
