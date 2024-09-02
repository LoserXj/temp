import transbigdata as tbd
import numpy as np
import hashlib
from scipy.spatial import KDTree
import osmnx as ox
# 定义环境状态的类
import time
from tqdm import tqdm
import random
import pandas as pd
from scipy.stats import fatiguelife
import gc
import abm.config as config 
import re 


class ChargingStationAgent:
    def __init__(self, station_id, lon, lat, max_capacity, charge_speed_station, step_length, unit_price, station_type, station_owner=None):
        self.station_id = station_id    # 充电站id
        self.lon = float(lon)                          # 经度
        self.lat = float(lat)                          # 纬度
        self.max_capacity = max_capacity       # 最大容量
        self.remaining_capacity = max_capacity  # 剩余容量
        # 该充电站最大充电速率 (kW)
        self.charge_speed_station = charge_speed_station
        self.waiting_queue = []                # 等待队列
        self.current_queue = []                # 正在充电的队列
        self.infos = []                         # 用于存储每一仿真步的信息 # 列名列表
        self.step_length = step_length        # 每一步的时间长度（分钟）
        self.unit_price = unit_price           # 充电单价    
        self.num_charge_car_come = 0
        self.num_charge_car_left = 0
        self.current_charge_speed = 0                  # 当前充电功率
        self.station_type = station_type
        self.station_owner = station_owner
        

    def save_infos(self, environment):
        """
        保存充电站在每一仿真步的信息，用于后续分析和处理。
        
        参数:
        - self: 指的是调用该方法的充电站对象实例。
        - environment: 指的是仿真环境对象，包含了仿真过程中的各种参数和状态信息。
        
        返回值:
        - 无。该方法直接更新仿真环境的station_info属性，不返回值。
        
        详细说明:
        - 该方法首先检查仿真环境对象environment中是否存在station_info属性，如果不存在，则初始化一个空列表。
        - 接着，检查充电站的当前充电队列(current_queue)和等待队列(waiting_queue)是否有车辆。
        - 如果存在正在充电或等待充电的车辆，将充电站的相关信息和状态保存到仿真环境的station_info列表中。
        - 保存的信息包括时间戳(timestamp)、仿真步数(step)、充电站ID(station_id)、经纬度(lon, lat)、最大容量(max_capacity)、剩余容量(remaining_capacity)、充电速度(charge_speed_station)、当前充电车辆列表(current_car)、等待充电车辆列表(waiting_car)、当前充电车辆数量(num_current_car)、等待充电车辆数量(num_waiting_car)、今日来到充电站的车辆数量(num_charge_car_come)、今日离开充电站的车辆数量(num_charge_car_left)、当前充电速度(current_charge_speed)、充电站类型(station_type)和充电站所有者(station_owner)。
        - 通过这个方法，可以记录充电站在仿真过程中的运行情况，为进一步的数据分析和充电站管理提供支持。
        """
        #判断environment中是否存在carinfos属性
        if not hasattr(environment, 'station_info'):
            environment.station_info = []

        # 存储每一仿真步的信息
        if (len(self.current_queue)>0)|(len(self.waiting_queue)>0):
            environment.station_info.append({
                'timestamp': environment.timestamp,
                'step': environment.step,
                'station_id': self.station_id,
                'lon': self.lon,
                'lat': self.lat,
                'max_capacity': self.max_capacity,
                'remaining_capacity': self.remaining_capacity,
                'charge_speed_station': self.charge_speed_station,
                'current_car': [current_car.carid for current_car in self.current_queue],
                'waiting_car': [current_car.carid for current_car in self.waiting_queue],
                'num_current_car': len(self.current_queue),
                'num_waiting_car': len(self.waiting_queue),
                'num_charge_car_come': self.num_charge_car_come,
                'num_charge_car_left': self.num_charge_car_left,
                'current_charge_speed': self.current_charge_speed,
                'station_type': self.station_type,
                'station_owner': self.station_owner                
            })        

    def if_not_exit_in_queues(self, car_agent):
        """
        判断车辆是否在当前队列或等待队列中。
        
        参数:
        - car_agent: 指的是车辆代理对象。
        
        返回值:
        - 返回一个布尔值，如果车辆不在当前队列和等待队列中，则为True；否则为False。
        """
        # 判断车辆是否在队列中
        return (car_agent not in self.current_queue) and (car_agent not in self.waiting_queue)

    def has_capacity(self):
        """
        判断充电站是否有空余位置。
        
        返回值:
        - 如果有空余位置，即当前队列中的车辆数量小于最大容量，则返回True；否则返回False。
        """
        # 判断是否有空余位置
        return len(self.current_queue) < self.max_capacity
    
    def capacity(self):
        """
        获取充电站的剩余容量。
        
        返回值:
        - 返回充电站的剩余容量，即最大容量减去当前队列中车辆的数量。
        """
        return self.max_capacity - len(self.current_queue)

    def charge_cars(self):
        """
        给当前队列中的车辆充电。
        
        详细说明:
        - 遍历当前队列中的每个车辆代理对象，调用车辆代理的charging_per_step方法进行充电。
        - 更新充电站的当前充电速度，为所有正在充电车辆的充电速度之和。
        """
        # 给current队列中的车辆充电
        self.current_charge_speed = 0
        for car_agent in self.current_queue:
            # 计算车辆充电速率
            car_charge_speed = car_agent.charging_per_step(self)
            car_agent.status = 2 # 设置为充电状态
            self.current_charge_speed += car_charge_speed
    
    def add_to_queue(self, car_agent):
        """
        将车辆加入到充电站的队列中。
        
        参数:
        - car_agent: 指的是车辆代理对象。
        
        详细说明:
        - 首先判断车辆是否已经在队列中，如果不在，则检查充电站是否有空余位置。
        - 如果有，则将车辆加入到当前队列中，并设置车辆状态为充电状态。
        - 如果没有空余位置，但等待队列未满，则将车辆加入到等待队列中，并设置车辆状态为等待状态。
        """
        # 将车辆加入到队列中
        if self.if_not_exit_in_queues(car_agent):
            if self.has_capacity():
                # 如果有空余位置就加入到current队列
                self.current_queue.append(car_agent)
                car_agent.status = 2
                self.num_charge_car_come +=1
            elif len(self.waiting_queue) < 999999:  # waiting队列最大长度
                # 如果没有空余位置就加入到waiting队列
                self.waiting_queue.append(car_agent)
                car_agent.status = 3   

    def move_cars_from_waiting_to_current(self):
        """
        将等待队列中的车辆加入到当前队列中。
        
        详细说明:
        - 当前队列有空余位置时，从等待队列中将车辆逐个移动到当前队列，直到等待队列为空或当前队列已满。
        """
        # 将waiting队列中的车辆加入到current队列中
        while self.has_capacity() and len(self.waiting_queue) > 0:  # 如果有空余位置并且waiting队列不为空

            # 从waiting队列中第一个车辆，加入到current队列
            car_agent = self.waiting_queue.pop(0)
            if car_agent not in self.current_queue:
                self.current_queue.append(car_agent)
                car_agent.status = 2  # 设置为充电状态
                self.num_charge_car_come +=1    

    def run(self, environment):
        """
        执行充电站在仿真环境中的运行逻辑。
        
        参数:
        - environment: 指的是仿真环境对象，包含了仿真过程中的各种参数和状态信息。
        
        详细说明:
        - 如果当前队列为空，设置充电站的当前充电速度为0。
        - 如果有车辆在当前队列或等待队列中，执行充电操作，并将等待队列中的车辆移动到当前队列。
        - 更新充电站的剩余容量，并保存充电站的信息至仿真环境。
        """
        # 充电站运行
        if len(self.current_queue) == 0:
            self.current_charge_speed = 0

        if (len(self.current_queue) > 0) | (len(self.waiting_queue) > 0):
            # 给current队列中的车充电
            self.charge_cars()
            # 如果有空余位置就把waiting队列pop 并加入到current队列
            self.move_cars_from_waiting_to_current()
            self.remaining_capacity = self.max_capacity - \
                len(self.current_queue)  # 更新剩余容量
            
        self.save_infos(environment)  # 保存信息