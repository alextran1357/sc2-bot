from pysc2.agents import base_agent
from pysc2.env import sc2_env, run_loop
from pysc2.lib import actions, features, units
from absl import app
import numpy as np
import pandas as pd
import os

# Possible Actions
ACTION_DO_NOTHING = 'donothing'
ACTION_BUILD_PROBE = 'buildprobe'
ACTION_BUILD_PYLON = 'buildpylon'
ACTION_BUILD_GATEWAY = 'buildgateway'
ACTION_BUILD_ZEALOT = 'buildzealot'
ACTION_ATTACK = 'attack'

# Define Reward
KILL_UNIT_REWARD = 0.2
KILL_STRUCTURE_REWARD = 0.5

smart_actions = [
   ACTION_DO_NOTHING,
   ACTION_BUILD_PROBE,
   ACTION_BUILD_PYLON,
   ACTION_BUILD_GATEWAY,
   ACTION_BUILD_ZEALOT,
   ACTION_ATTACK,
]

class QLearningTable():
   def __init__(self, actions, learning_rate=0.01, gamma=0.9, epsilon=0.9, pickle_file=None):
      self.actions = actions
      self.learning_rate = learning_rate
      self.gamma = gamma
      self.epsilon = epsilon
      
      if pickle_file == None:
         self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
      else:
         self.q_table = pd.read_pickle('q_table.pkl')
      
   def choose_action(self, observation):
      self.check_state_exist(observation)
      
      if np.random.uniform() > self.epsilon:
         action = np.random.choice(self.actions)
         
      else:
         state_action = self.q_table.loc[observation]
         state_action = state_action.reindex(np.random.permutation(state_action.index))
         action = state_action.idxmax()
      
      return action

   def check_state_exist(self, state):
      if state not in self.q_table.index: 
         self.q_table = pd.concat([self.q_table,pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state).to_frame().T])
   
   def learn(self, state, action, reward, next_state):
      self.check_state_exist(state)
      self.check_state_exist(next_state)
      
      q_predict = self.q_table.loc[state, action]
      q_target = reward + self.gamma * self.q_table.loc[next_state, :].max()
      
      self.q_table.loc[state, action] += self.learning_rate * (q_target - q_predict)
   
class ProtossAgent(base_agent.BaseAgent):
   def __init__(self):
      super(ProtossAgent, self).__init__()
      self.base_top_left = False
      
      # Check pickle q_table exist
      path = './q_table.pkl'
      if os.path.isfile(path):
         self.q_table = QLearningTable(actions=list(range(len(smart_actions))),pickle_file=path)
      else:
         self.q_table = QLearningTable(actions=list(range(len(smart_actions))))
      
      self.previous_killed_unit_score = 0
      self.previous_killed_stucture_score = 0
      
      self.previous_action = None
      self.previous_state = None

   # HELPER FUNCTIONS ----------------------------------------------------------
   def unit_type_is_selected(self, obs, unit_type):
      if (len(obs.observation.single_select) > 0 and obs.observation.single_select[0].unit_type == unit_type):
         return True
       
      if (len(obs.observation.multi_select) > 0 and obs.observation.multi_select[0].unit_type == unit_type):
         return True
      
      return False

   def get_units_by_type(self, obs, unit_type):
      return [unit for unit in obs.observation.raw_units
            if unit.unit_type == unit_type 
            and unit.alliance == features.PlayerRelative.SELF]
   
   def get_my_completed_units_by_type(self, obs, unit_type):
    return [unit for unit in obs.observation.raw_units
            if unit.unit_type == unit_type 
            and unit.build_progress == 100
            and unit.alliance == features.PlayerRelative.SELF]

   def can_do(self, obs, action):
      return action in obs.observation.available_actions
   
   def get_distances(self, units, buildingxy):
      units_xy = [(unit.x, unit.y) for unit in  units]
      return np.linalg.norm(np.array(units_xy) - np.array(buildingxy), axis=1)
      
   def check_available_supply(self, obs):
      return obs.observation.player.food_cap - obs.observation.player.food_used

   def generate_random_xy_position(self, pylon_position=None):
      top_left_low = 5
      top_left_high = 25
      bottom_right_low = 39
      bottom_right_high = 59
      low_x = 0
      high_x = 0
      low_y = 0
      high_y = 0
      if pylon_position: # build around pylon radius
         low_x = np.random.randint(low=-3, high=2)
         high_x = np.random.randint(low=-3, high=2)
         low_y = np.random.randint(low=-3, high=2)
         high_y = np.random.randint(low=-3, high=2)
      if self.base_top_left:  
         x_position = np.random.randint(low=top_left_low+low_x, high=top_left_high+high_x)
         y_position = np.random.randint(low=top_left_low+low_y, high=top_left_high+high_y)
      else:
         x_position = np.random.randint(low=bottom_right_low+low_x, high=bottom_right_high+high_x)
         y_position = np.random.randint(low=bottom_right_low+low_y, high=bottom_right_high+high_y)
      return (x_position, y_position)
   
   # def action_idle_worker_mine(self, obs):
   #    probes = self.get_my_units_by_type(obs, units.Protoss.Probe)
   #    idle_probes = [probe for probe in probes if probe.order_length == 0]
   #    if len(idle_probes) > 0:
   #       mineral_patches = [unit for unit in obs.observation.raw_units
   #                         if unit.unit_type in [
   #                            units.Neutral.BattleStationMineralField,
   #                            units.Neutral.BattleStationMineralField750,
   #                            units.Neutral.LabMineralField,
   #                            units.Neutral.LabMineralField750,
   #                            units.Neutral.MineralField,
   #                            units.Neutral.MineralField750,
   #                            units.Neutral.PurifierMineralField,
   #                            units.Neutral.PurifierMineralField750,
   #                            units.Neutral.PurifierRichMineralField,
   #                            units.Neutral.PurifierRichMineralField750,
   #                            units.Neutral.RichMineralField,
   #                            units.Neutral.RichMineralField750
   #                         ]]
   #       probe = random.choice(idle_probes)
   #       distances = self.get_distances(mineral_patches, (probe.x, probe.y))
   #       mineral_patch = mineral_patches[np.argmin(distances)] 
   #       return actions.RAW_FUNCTIONS.Harvest_Gather_unit("now", probe.tag, mineral_patch.tag)

   # STEP FUNCTION ----------------------------------------------------------------
   def step(self, obs):
      super(ProtossAgent, self).step(obs)
      
      # Getting position of self and the enemy
      if obs.first():
         nexus = self.get_units_by_type(obs, units.Protoss.Nexus)[0]
         self.base_top_left = (nexus.x < 32)
         
      if obs.last():
         self.q_table.q_table.to_pickle('q_table.pkl')
         self.q_table.q_table.to_csv('q_table.csv') # for visualization
         
      # Getting column values
      # number zealots
      zealot_count = len(self.get_units_by_type(obs, units.Protoss.Zealot))
      # number gateways
      gateway_count = len(self.get_units_by_type(obs, units.Protoss.Gateway))
      # number pylons
      pylon_count = len(self.get_units_by_type(obs, units.Protoss.Pylon))
      # army count
      army_supply = obs.observation.player.food_army
      # worker count
      worker_supply = obs.observation.player.food_workers

      killed_unit_score = obs.observation['score_cumulative'][5]
      killed_structure_score = obs.observation['score_cumulative'][6]

      current_state = [
         zealot_count,
         gateway_count,
         pylon_count,
         army_supply,
         worker_supply,
      ]

      # define score
      if self.previous_action is not None:
         reward = 0
         
         if killed_unit_score > self.previous_killed_unit_score:
            reward += KILL_UNIT_REWARD

         if killed_structure_score > self.previous_killed_structure_score:
            reward += KILL_STRUCTURE_REWARD
            
         self.q_table.learn(str(self.previous_state), self.previous_action, reward, str(current_state))
         
      rl_action = self.q_table.choose_action(str(current_state))
      smart_action = smart_actions[rl_action]
      
      self.previous_killed_unit_score = killed_unit_score
      self.previous_killed_structure_score = killed_structure_score
      self.previous_state = current_state
      self.previous_action = rl_action


      if smart_action == ACTION_DO_NOTHING:
         return actions.RAW_FUNCTIONS.no_op()

      elif smart_action == ACTION_BUILD_PROBE:
         available_nexuses = self.get_my_completed_units_by_type(obs, units.Protoss.Nexus)
         if (len(available_nexuses) > 0 
            and obs.observation.player.minerals > 50):
            nexus = available_nexuses[np.random.choice(list(range(len(available_nexuses))))]
            if nexus.order_length < 5:
               return actions.RAW_FUNCTIONS.Train_Probe_quick("now", nexus.tag)
      
      elif smart_action == ACTION_BUILD_PYLON:
         if (obs.observation.player.minerals >= 100 and self.check_available_supply(obs)<10):
            probes = self.get_units_by_type(obs, units.Protoss.Probe)
            if len(probes) > 0:
               pylon_xy = self.generate_random_xy_position()
               distances = self.get_distances(probes, pylon_xy)
               probe = probes[np.argmin(distances)]
               return actions.RAW_FUNCTIONS.Build_Pylon_pt("now", probe.tag, pylon_xy)
      
      elif smart_action == ACTION_BUILD_GATEWAY:
         available_pylones = self.get_my_completed_units_by_type(obs, units.Protoss.Pylon)
         if (len(available_pylones) > 0 and obs.observation.player.minerals >= 150):
            probes = self.get_units_by_type(obs, units.Protoss.Probe)
            if len(probes) > 0:
               pylon = available_pylones[np.random.choice(list(range(len(available_pylones))))]
               pylon_xy = (pylon.x, pylon.y)
               gateway_xy = self.generate_random_xy_position(pylon_position=pylon_xy)
               distances = self.get_distances(probes, gateway_xy)
               probe = probes[np.argmin(distances)]
               return actions.RAW_FUNCTIONS.Build_Gateway_pt("now", probe.tag, gateway_xy)
      
      elif smart_action == ACTION_BUILD_ZEALOT:
         available_gateways = self.get_my_completed_units_by_type(obs, units.Protoss.Gateway)
         if (len(available_gateways) > 0 
            and obs.observation.player.minerals > 100):
            gateway = available_gateways[np.random.choice(list(range(len(available_gateways))))]
            if gateway.order_length < 5:
               return actions.RAW_FUNCTIONS.Train_Zealot_quick("now", gateway.tag)
      
      elif smart_action == ACTION_ATTACK:
         zealots = self.get_units_by_type(obs, units.Protoss.Zealot)
         if len(zealots) > 0:
            attack_xy = (38, 44) if self.base_top_left else (19, 23)
            zealot_tags = [unit.tag for unit in zealots]
            return actions.RAW_FUNCTIONS.Attack_pt("now", zealot_tags, attack_xy)

      return actions.RAW_FUNCTIONS.no_op()
      
def main(unused_arg):
   agent = ProtossAgent()
   try:
      while True:
         with sc2_env.SC2Env(
            map_name="Simple64",
            players=[sc2_env.Agent(sc2_env.Race.protoss), 
                     sc2_env.Bot(sc2_env.Race.protoss, 
                                 sc2_env.Difficulty.medium)],
            agent_interface_format=features.AgentInterfaceFormat(
               action_space=actions.ActionSpace.RAW,
               use_raw_units=True,
               raw_resolution=64,
            ),
         ) as env:
            run_loop.run_loop([agent], env)
            
   except KeyboardInterrupt:
      pass

  
if __name__ == "__main__":
  app.run(main)