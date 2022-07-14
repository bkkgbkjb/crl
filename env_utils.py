# coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility for loading the goal-conditioned environments."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import ant_env
import fetch_envs
import gym
# import metaworld
import numpy as np
import point_env

os.environ['SDL_VIDEODRIVER'] = 'dummy'


def euler2quat(euler):
  """Convert Euler angles to quaternions."""
  euler = np.asarray(euler, dtype=np.float64)
  assert euler.shape[-1] == 3, 'Invalid shape euler {}'.format(euler)

  ai, aj, ak = euler[Ellipsis, 2] / 2, -euler[Ellipsis, 1] / 2, euler[Ellipsis, 0] / 2
  si, sj, sk = np.sin(ai), np.sin(aj), np.sin(ak)
  ci, cj, ck = np.cos(ai), np.cos(aj), np.cos(ak)
  cc, cs = ci * ck, ci * sk
  sc, ss = si * ck, si * sk

  quat = np.empty(euler.shape[:-1] + (4,), dtype=np.float64)
  quat[Ellipsis, 0] = cj * cc + sj * ss
  quat[Ellipsis, 3] = cj * sc - sj * cs
  quat[Ellipsis, 2] = -(cj * ss + sj * cc)
  quat[Ellipsis, 1] = cj * cs - sj * sc
  return quat


def load(env_name):
  """Loads the train and eval environments, as well as the obs_dim."""
  # pylint: disable=invalid-name
  kwargs = {}
  if env_name == 'sawyer_push':
    max_episode_steps = 150
  elif env_name == 'sawyer_drawer':
    max_episode_steps = 150
  elif env_name == 'sawyer_drawer_image':
    max_episode_steps = 50
    kwargs['task'] = 'openclose'
  elif env_name == 'sawyer_window_image':
    kwargs['task'] = 'openclose'
    max_episode_steps = 50
  elif env_name == 'sawyer_push_image':
    max_episode_steps = 150
    kwargs['start_at_obj'] = True
  elif env_name == 'sawyer_bin':
    max_episode_steps = 150
  elif env_name == 'sawyer_bin_image':
    max_episode_steps = 150
  elif env_name == 'sawyer_window':
    max_episode_steps = 150
  elif env_name == 'fetch_reach':
    CLASS = fetch_envs.FetchReachEnv
    max_episode_steps = 50
  elif env_name == 'fetch_push':
    CLASS = fetch_envs.FetchPushEnv
    max_episode_steps = 50
  elif env_name == 'fetch_reach_image':
    CLASS = fetch_envs.FetchReachImage
    max_episode_steps = 50
  elif env_name == 'fetch_push_image':
    CLASS = fetch_envs.FetchPushImage
    max_episode_steps = 50
    kwargs['rand_y'] = True
  elif env_name.startswith('ant_'):
    _, map_name = env_name.split('_')
    assert map_name in ['umaze', 'medium', 'large']
    CLASS = ant_env.AntMaze
    kwargs['map_name'] = map_name
    kwargs['non_zero_reset'] = True
    if map_name == 'umaze':
      max_episode_steps = 700
    else:
      max_episode_steps = 1000
  elif env_name.startswith('offline_ant'):
    CLASS = lambda: ant_env.make_offline_ant(env_name)
    if 'umaze' in env_name:
      max_episode_steps = 700
    else:
      max_episode_steps = 1000
  elif env_name.startswith('point_image'):
    CLASS = point_env.PointImage
    kwargs['walls'] = env_name.split('_')[-1]
    if '11x11' in env_name:
      max_episode_steps = 100
    else:
      max_episode_steps = 50
  elif env_name.startswith('point_'):
    CLASS = point_env.PointEnv
    kwargs['walls'] = env_name.split('_')[-1]
    if '11x11' in env_name:
      max_episode_steps = 100
    else:
      max_episode_steps = 50
  else:
    raise NotImplementedError('Unsupported environment: %s' % env_name)

  # Disable type checking in line below because different environments have
  # different kwargs, which pytype doesn't reason about.
  gym_env = CLASS(**kwargs)  # pytype: disable=wrong-keyword-args
  obs_dim = gym_env.observation_space.shape[0] // 2
  return gym_env, obs_dim, max_episode_steps

