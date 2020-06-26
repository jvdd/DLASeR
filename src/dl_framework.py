
from dataclasses import dataclass
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from src.utils import Operator, operator_to_thresh_str, operator_to_opt_str

@dataclass
class ThresholdGoal:
    quality: str
    operator: Operator
    value: float

@dataclass
class SetpointGoal:
    quality: str
    value: float
    epsilon: float

@dataclass
class OptimizationGoal:
    quality: str
    operator: Operator

class DLFramework:

    def __init__(self, qualities):
        self.qualities = qualities
        self.threshold_goals = {}
        self.setpoint_goals = {}
        self.optimization_goals = {}
        self.callback = [EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                         ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.3, min_lr=0.0001)]


    @staticmethod
    def _get_threshold_name(quality, threshold_value):
        name = quality+'_thresh'+str(threshold_value)
        return name

    @staticmethod
    def _get_setpoint_name(quality, setpoint_value):
        name = quality+'_setpoint'+str(setpoint_value)
        return name

    @staticmethod
    def _get_optimization_name(quality):
        name = quality+'_opt'
        return name


    def add_threshold_goal(self, quality, operator, threshold_value):
        assert quality in self.qualities # Must be a known quality
        threshold_goal = ThresholdGoal(quality, operator, threshold_value)
        name = self._get_threshold_name(quality, threshold_value)
        self.threshold_goals[name] = threshold_goal

    def add_setpoint_goal(self, quality, setpoint_value, epsilon):
        assert quality in self.qualities # Must be a known quality
        setpoint_goal = SetpointGoal(quality, setpoint_value, epsilon)
        name = self._get_setpoint_name(quality, setpoint_value)
        self.setpoint_goals[name] = setpoint_goal

    def add_optimization_goal(self, quality, operator):
        assert quality in self.qualities # Must be a known quality
        assert len(self.optimization_goals) == 0 # We do not support multi-optimization goals
        optimization_goal = OptimizationGoal(quality, operator)
        name = self._get_optimization_name(quality)
        self.optimization_goals[name] = optimization_goal


    def print_goals(self):
        if len(self.threshold_goals) > 0:
            print('-'*10+' THRESHOLD GOALS '+10*'-')
            for threshold_goal in self.threshold_goals.values():
                print(threshold_goal.quality, operator_to_thresh_str(threshold_goal.operator), threshold_goal.value)
        if len(self.setpoint_goals) > 0:
            print('-'*10+' SET-POINT GOALS '+10*'-')
            for setpoint_goal in self.setpoint_goals.values():
                print(setpoint_goal.quality, '==', setpoint_goal.value)
        if len(self.optimization_goals) > 0:
            print('-'*9+' OPTIMIZATION GOALS '+9*'-')
            for optimization_goal in self.optimization_goals.values():
                print(operator_to_opt_str(optimization_goal.operator), optimization_goal.quality)