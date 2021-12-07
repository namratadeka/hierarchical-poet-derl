import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from os.path import join

import wandb
import torch
from torch.distributions import MultivariateNormal

from poet_distributed.niches.box2d.actor_critic import Actor_Critic
from poet_distributed.niches.box2d.bipedal_walker_custom import BipedalWalkerCustom, Env_config


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
    

class PPO:
    def __init__(self, 
                 env_config,
                 morph_params, 
                 model_dir='',
                 decay_lr=True,
                 lr_end_factor=0.1,
                 lr_decay_iters=400,
                 seed=0,
                 mode=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._init_hyperparameters()
        self.mode = mode

        self.decay_lr = decay_lr
        self.lr_decay_iters = lr_decay_iters
        self.lr_end_factor = lr_end_factor

        self._build_network()
        self._build_optimizer()

        self.env = BipedalWalkerCustom(env_config)
        self.env.seed(seed)
        self.env.set_env_config(env_config)
        self.set_morph_params(morph_params)
        self.act_dim = self.env.action_space.shape[0]
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var).to(self.device)

        self.start_score = None
        self.score = None
        self.prev_score = -np.inf

        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

    def _init_hyperparameters(self):
        self.timesteps_per_batch = 2048
        self.max_timesteps_per_episode = 1024
        self.gamma = 0.9
        self.epochs = 7
        self.clip = 0.2
        self.entropy_beta = 0.01
        self.minibatch_size = 256
        self.base_lr = 3e-4
        self.value_coeff = 0.5
        self.max_grad_norm = 0.5
        self.skip_frames = 4


    def set_morph_params(self, morph_params):
        self.morph_params = morph_params
        self.env.set_morphology(morph_params)
        self.morph_id = 'm_%s'%'_'.join([str(x) for x in morph_params])
        
    def _build_network(self):
        self.actor_critic = Actor_Critic().to(self.device)
        self.actor_critic.apply(initialize_weights)

        if self.mode == 'train':
            wandb.watch(self.actor_critic)

    def _build_optimizer(self):
        params =list(self.actor_critic.parameters())
        self.optim = torch.optim.Adam(
            params, lr=self.base_lr,
            eps=1e-5
        )

        if self.decay_lr:
            self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optim,
                start_factor=1, 
                end_factor=self.lr_end_factor,
                total_iters=self.lr_decay_iters
            )
    
    def get_action(self, state, actor:Actor_Critic):
        mean, _ = actor(state)
        mean = mean.squeeze()
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
            state = self.env.reset()
            done = False

            for ep_t in range(self.max_timesteps_per_episode):
                if render:
                    self.env.render()
                t += 1
                batch_states.append(state)
                action, log_prob = self.get_action(state, self.actor_critic)
                for k in range(self.skip_frames):
                    state, rew, done, _ = self.env.step(action)
                
                ep_rews.append(rew)
                batch_actions.append(action)
                batch_logprobs.append(log_prob)

                if done:
                    break
            
            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)

        batch_states = torch.tensor(np.array(batch_states), dtype=torch.float).to(self.device)
        batch_actions = torch.tensor(np.array(batch_actions), dtype=torch.float).to(self.device)
        batch_logprobs = torch.tensor(batch_logprobs).to(self.device)

        batch_rtgs = self.compute_rtgs(batch_rews)

        return batch_states, batch_actions, batch_logprobs, batch_rtgs, batch_lens

    def eval_agent(self, actor:Actor_Critic=None, num_episodes:int=5, max_episode_length:int=2000):
        '''
        Evaluate agent over multiple episodes and return average total reward.
        '''
        if actor is None:
            actor = self.actor_critic
        actor.eval()
        reward_list = []
        for i in tqdm(range(num_episodes)):
            total_reward = 0
            state = self.env.reset()
            t = 0
            while t < max_episode_length:
                t += 1
                action, _ = self.get_action(state, actor)
                state, reward, done, _ = self.env.step(action)
                total_reward += reward
                if done:
                    break
            reward_list.append(total_reward)
        return reward_list

    def update_score(self):
        self.score = np.mean(self.eval_agent())

    def evaluate_critic(self, states, actions):
        mean, V = self.actor_critic(states)
        mean, V = mean.squeeze(), V.squeeze()
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(actions)

        return V, log_probs

    def learn(self, total_timesteps=100000, render=False, iteration=0):
        self.actor_critic.train()
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
                    #normalize advantages over batch
                    A_k = (A_k - A_k.mean()) / (A_k.std(unbiased=False) + 1e-5)

                    ratios = torch.exp(curr_logprobs - log_probs)

                    #surrogate losses
                    surr_1 = ratios * A_k
                    surr_2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

                    entropy = self.entropy_beta * (-(torch.exp(curr_logprobs) * curr_logprobs)).mean()

                    actor_loss = (-torch.min(surr_1, surr_2)).mean() - entropy
                    critic_loss = torch.nn.MSELoss()(V, rtgs)

                    loss = self.value_coeff*critic_loss + actor_loss
                    self.optim.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.actor_critic.parameters(), self.max_grad_norm
                    )
                    self.optim.step()
            
            if self.decay_lr:
                self.lr_scheduler.step()
            
            itr += 1
            self.update_score()
            wandb.log({'Avg. reward over 5 episodes':self.score}, step=itr)
            if self.score > self.prev_score:
                self.prev_score = self.score
                self.save(itr)
            elif itr % 10 == 0:
                self.save(itr)
        
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

    def save(self, itr):
        self.actor_critic.eval()
        save_dict = {
            'iteration': itr,
            'opt': self.optim.state_dict(),
            'actor_critic': self.actor_critic.state_dict(),
            'env': self.env.config.name,
            'env_config': self.env.config._asdict(),
            'morph_params': self.morph_params,
            'score': self.score,
        }
        model_path = join(self.model_dir, '{}_{}.pth'.format(itr, self.score))
        torch.save(save_dict, model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train PPO from scratch given an environment and morphology config.'
    )
    parser.add_argument(
        '--env',
        required=True,
        type=str,
        help='path to json containing env config'
    )
    parser.add_argument(
        '--morph',
        default='1_1_1_1_1_1_1_1',
        type=str,
        help='morph config string'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='train'
    )
    parser.add_argument(
        '--load',
        type=str,
        default=None,
        help='required if mode != train'
    )
    (args, unknown_args) = parser.parse_known_args()
    morph_params = [float(x) for x in args.morph.split('_')]
    env = json.load(open(args.env, 'r'))
    env_config = Env_config(
        name=env['config']['name'],
        ground_roughness=env['config']['ground_roughness'],
        pit_gap=env['config']['pit_gap'],
        stump_width=env['config']['stump_width'],
        stump_height=env['config']['stump_height'],
        stump_float=env['config']['stump_float'],
        stair_height=env['config']['stair_height'],
        stair_width=env['config']['stair_width'],
        stair_steps=env['config']['stair_steps']
    )

    agent = PPO(
            env_config=env_config, 
            morph_params=morph_params,
            model_dir=join('./models', env_config.name, args.morph),
            decay_lr=True,
            seed=env['seed'],
            mode=args.mode)
    if args.mode == 'train':
        wandb.init(project="532j-project", name='{}_m_{}'.format(env_config.name, args.morph), config=env['config'])
        agent.learn(total_timesteps=1000000)
    else:
        print("Evaluating agent ..")
        assert args.load is not None
        weights = torch.load(args.load)
        agent.actor_critic.load_state_dict(weights['actor_critic'])
        rewards = agent.eval_agent(num_episodes=50)
        np.save('{}_m_{}.npy'.format(env_config.name, args.morph), rewards)
