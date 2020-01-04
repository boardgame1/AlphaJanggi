import numpy, random
import collections
from typing import List, Optional

from environ import Environment, Player, MAX_TURN
from model import Action, ActionHistory, Network
from actionTable import AllMoveLength

MAXIMUM_FLOAT_VALUE = float('inf')

KnownBounds = collections.namedtuple('KnownBounds', ['min', 'max'])

class MinMaxStats(object):
  """A class that holds the min-max values of the tree."""

  def __init__(self, known_bounds: Optional[KnownBounds]):
    self.maximum = known_bounds.max if known_bounds else -MAXIMUM_FLOAT_VALUE
    self.minimum = known_bounds.min if known_bounds else MAXIMUM_FLOAT_VALUE

  def update(self, value: float):
    self.maximum = max(self.maximum, value)
    self.minimum = min(self.minimum, value)

  def normalize(self, value: float) -> float:
    if self.maximum > self.minimum:
      # We normalize only when we have set the maximum and minimum values.
      return (value - self.minimum) / (self.maximum - self.minimum)
    return value


class MuZeroConfig(object):

  def __init__(self,
               action_space_size: int,
               max_moves: int,
               discount: float,
               dirichlet_alpha: float,
               num_simulations: int,
               batch_size: int,
               td_steps: int,
               lr_init: float,
               lr_decay_steps: float,
               visit_softmax_temperature_fn,
               known_bounds: Optional[KnownBounds] = None):
    ### Self-Play
    self.action_space_size = action_space_size

    self.visit_softmax_temperature_fn = visit_softmax_temperature_fn
    self.max_moves = max_moves
    self.num_simulations = num_simulations
    self.discount = discount

    # Root prior exploration noise.
    self.root_dirichlet_alpha = dirichlet_alpha
    self.root_exploration_fraction = 0.25

    # UCB formula
    self.pb_c_base = 19652
    self.pb_c_init = 1.25

    # If we already have some information about which values occur in the
    # environment, we can use them to initialize the rescaling.
    # This is not strictly necessary, but establishes identical behaviour to
    # AlphaZero in board games.
    self.known_bounds = known_bounds

    ### Training
    self.checkpoint_interval = int(10)
    self.window_size = int(10)
    self.batch_size = batch_size
    self.num_unroll_steps = 4
    self.td_steps = td_steps

    self.weight_decay = 1e-4
    self.momentum = 0.9

    # Exponential learning rate schedule
    self.lr_init = lr_init
    self.lr_decay_rate = 0.1
    self.lr_decay_steps = lr_decay_steps

  def new_game(self):
    return Game(self.action_space_size, self.discount)


def make_board_game_config(action_space_size: int, max_moves: int,
                           dirichlet_alpha: float,
                           lr_init: float) -> MuZeroConfig:

  def visit_softmax_temperature(num_moves):
    if num_moves < 30:
      return 1.0
    else:
      return 0.0  # Play according to the max.

  return MuZeroConfig(
      action_space_size=action_space_size,
      max_moves=max_moves,
      discount=1.0,
      dirichlet_alpha=dirichlet_alpha,
      num_simulations=10,
      batch_size=64,
      td_steps=max_moves,  # Always use Monte Carlo return.
      lr_init=lr_init,
      lr_decay_steps=400e3,
      visit_softmax_temperature_fn=visit_softmax_temperature,
      known_bounds=KnownBounds(-1, 1))

def make_janggi_config() -> MuZeroConfig:
  return make_board_game_config(
      action_space_size=AllMoveLength, max_moves=MAX_TURN, dirichlet_alpha=0.17, lr_init=0.01)

class Node(object):

  def __init__(self, prior: float):
    self.visit_count = 0
    self.to_play = -1
    self.prior = prior
    self.value_sum = 0
    self.children = {}
    self.hidden_state = None
    self.reward = 0

  def expanded(self) -> bool:
    return len(self.children) > 0

  def value(self) -> float:
    if self.visit_count == 0:
      return 0
    return self.value_sum / self.visit_count


class Game(object):
  """A single episode of interaction with the environment."""

  def __init__(self, action_space_size: int, discount: float):
    self.environment = Environment().reset()  # Game specific environment.
    self.history = []
    self.rewards = []
    self.child_visits = []
    self.root_values = []
    self.action_space_size = action_space_size
    self.discount = discount

  def terminal(self) -> bool:
    # Game specific termination rules.
    return self.environment.done

  def legal_actions(self) -> List[Action]:
    # Game specific calculation of legal actions.
    return self.environment.legal_actions()

  def apply(self, action: Action):
    reward = self.environment.step(action)
    reward = reward if self.environment.turn % 2 != 0 and reward == 1 else -reward
    self.rewards.append(reward)
    self.history.append(action)

  def store_search_statistics(self, root: Node):
    sum_visits = sum(child.visit_count for child in root.children.values())
    cv = []
    for a in root.children:
      cv.append(root.children[a].visit_count / sum_visits)
      cv.append(a)
    self.child_visits.append(cv)
    self.root_values.append(root.value())

  def make_image(self, state_index: int):
    # Game specific feature planes.
    o = Environment().reset()

    for current_index in range(0, state_index):
      o.step(self.history[current_index])

    return o.black_and_white_plane()

  def make_target(self, state_index: int, num_unroll_steps: int, td_steps: int):
    # The value target is the discounted root value of the search tree N steps
    # into the future, plus the discounted sum of all rewards until then.
    targets = []
    for current_index in range(state_index, state_index + num_unroll_steps + 1):
      bootstrap_index = current_index + td_steps
      if bootstrap_index < len(self.root_values):
        value = self.root_values[bootstrap_index] * self.discount**td_steps
      else:
        value = 0

      for i, reward in enumerate(self.rewards[current_index:bootstrap_index]):
        value += reward * self.discount**i  # pytype: disable=unsupported-operands

      if current_index < len(self.root_values):
        cv = [0.0]*self.action_space_size
        for i in range(0, len(self.child_visits[current_index]), 2):
          cv[self.child_visits[current_index][i+1]] = self.child_visits[current_index][i]
        targets.append((value, self.rewards[current_index], cv))
      else:
        # States past the end of games are treated as absorbing states.
        targets.append((0, 0, []))
    return targets

  def to_play(self) -> Player:
    return self.environment.player_turn

  def action_history(self) -> ActionHistory:
    return ActionHistory(self.history, self.action_space_size)

class ReplayBuffer(object):

  def __init__(self, config: MuZeroConfig):
    self.window_size = config.window_size
    self.batch_size = config.batch_size
    self.buffer = []

  def save_game(self, game):
    if len(self.buffer) > self.window_size:
      self.buffer.pop(0)
    self.buffer.append(game)

  def sample_batch(self, num_unroll_steps: int, td_steps: int):
    games = [self.sample_game() for _ in range(self.batch_size)]
    game_pos = [(g, self.sample_position(g)) for g in games]
    return [(g.make_image(i), g.history[i:i + num_unroll_steps],
             g.make_target(i, num_unroll_steps, td_steps))
            for (g, i) in game_pos]

  def sample_game(self) -> Game:
    # Sample game from buffer either uniformly or according to some priority.
    return numpy.random.choice(self.buffer)

  def sample_position(self, game) -> int:
    # Sample position from game either uniformly or according to some priority.
    return random.randrange(0, len(game.history))
