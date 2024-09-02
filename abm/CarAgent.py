import transbigdata as tbd
import numpy as np
# 定义环境状态的类
import time
import random
import abm.config as config
from datetime import datetime, timedelta



class CarAgent:
    def __init__(
            self,
            carid,
            battery,
            init_lon,
            init_lat,
            tripchain,
            step_length,
            vehicle_power_type,
            vehicle_type,
            total_power,
            battery_type,
            charge_decision_power,
            has_private_pile,                      
            consumption_rate=20,
            charge_speed_car=20,
            travel_speed = 40
    ):
        self.carid = carid                      # 车辆ID
        self.battery = battery                  # 剩余电量kWh
        self.status = 0                         # 0 停车，1 行驶，2 充电中，3 充电排队中
        self.lon = init_lon                     # 经度
        self.lat = init_lat                     # 纬度
        self.position = {}                        # 所在边信息
        self.total_power = total_power            # 总电量（kwh）
        self.battery_type = battery_type          # 电池类型，0 low capacity， 1 mid capacity， 2 high capacity
        self.consumption_rate = consumption_rate  # 百公里耗电（kwh）
        self.charge_speed_car = charge_speed_car  # 该车的最大充电速度（kwh/h）
        self.current_charge_speed = 0             # 当前充电速度
        self.tripchain = tripchain                # 行程链
        self.infos = []                           # 用于存储每一仿真步的信息
        self.step_length = step_length            # 每一步的时间长度（分钟）
        self.vehicle_power_type = vehicle_power_type #车辆动力类型
        self.vehicle_type = vehicle_type             # 车辆类型
        self.charge_station = ''                  # 当前所在充电站ID
        self.current_route = {}                   # 当前行驶路径
        self.current_route_traveldistance = 0     # 当前行驶路径的行驶距离，用于直线路径
        self.current_route_traveltime = 0         # 当前行驶时间
        self.target_charge_station = ''           # 当前希望前往的充电站ID
        self.travel_speed = travel_speed          # 行驶速度 km/h
        self.wether_home = 0
        self.wether_work = 0
        self.charge_infos = []
        self.charge_decision_power = charge_decision_power 
        self.has_private_pile = has_private_pile                     
                # 目标充电站坐标
        self.target_lon = None #----------
        self.target_lat = None #----------

    def save_infos(self, environment):
        if not hasattr(environment, 'car_info'):
            environment.car_info = []
        # 存储每一仿真步的信息
        if (not environment.save_car_park_infos) & (self.status==0):
            return None
        else:
            environment.car_info.append({
                'vehicle_power_type': self.vehicle_power_type,
                'vehicle_type': self.vehicle_type,
                'timestamp': environment.timestamp,
                'step': environment.step,
                'carid': self.carid,
                'battery': self.battery,
                'status': self.status,
                'lon': self.lon,
                'lat': self.lat,
                'total_power': self.total_power,
                'consumption_rate': self.consumption_rate,
                'soc': int(100*self.battery / self.total_power),# soc剩余电量百分比
                'current_charge_speed': self.current_charge_speed
            })

     # region 车辆：行驶
    def traveling_per_step(self, environment):
        """
        模拟车辆在仿真环境中每一步的行驶行为。如果车辆到达目的地，则返回True；否则返回False。
        
        参数:
        - self: 指的是调用该方法的车辆对象实例。
        - environment: 指的是仿真环境对象，包含了仿真过程中的各种参数和状态信息。
        
        返回值:
        - 如果车辆到达目的地，则返回True。
        - 如果车辆未到达目的地，继续行驶或等待，则返回False。
        
        详细说明:
        - 首先检查车辆是否处于充电状态或行驶状态，如果是，则停止充电。
        - 如果车辆电量为0，则将车辆状态设置为停车状态，并结束行驶。
        - 设置车辆状态为行驶状态，并根据当前路线信息进行行驶模拟。
        - 如果当前路线是直接行驶（即没有复杂的路径变化），则计算行驶距离，并更新车辆的行驶距离、消耗电量。
            - 如果行驶距离超过了路线长度，表示已经到达目的地，更新车辆位置，并清理路线信息，返回True。
            - 如果未到达目的地，根据行驶距离更新车辆的经纬度位置，并返回False。
        - 如果当前路线不是直接行驶，首先检查是否有可用路径。
            - 如果没有路径，表示可以直接到达目的地，消耗相应电量，并更新车辆位置，返回True。
            - 如果有路径，更新行驶时间，并根据当前位置和目标位置计算行驶距离，消耗电量。
            - 如果到达终点，更新车辆位置，并清理路线信息，返回True。
            - 如果未到达终点，更新车辆在路径上的位置，并返回False。
        """
        # 每一step的行驶行为，如果到达终点，返回True，否则返回False

        if (self.status == 2) | (self.status == 3):
            self.charging_stop(environment)  # 如果在充电，则停止充电
        
        #如果电量为0，则停止行动
        if self.battery<=0:
            self.status = 0  #设置为停车状态
            return 
        
        self.status = 1  # 设置为行驶状态
        if self.current_route['direct']:
            # 出行的距离
            travel_distance = self.travel_speed * 1000 * self.step_length / 60
            # 更新行驶距离
            self.current_route_traveldistance += travel_distance
            # 消耗电量
            self.consume_power(travel_distance)
            # 判断是否到达终点
            if self.current_route_traveldistance >= self.current_route['length']:
                self.lon = self.current_route['elon']
                self.lat = self.current_route['elat']

                # 清空路径信息
                self.current_route = {}
                self.status = 0                     # 设置为停车状态
                self.current_route_traveltime = 0
                self.current_route_traveldistance = 0
                return True
            else:
                # 更新经纬度位置
                current_route = self.current_route

                self.lon = (self.current_route_traveldistance / current_route['length']) * (
                    current_route['elon'] - current_route['slon']) + current_route['slon']
                self.lat = (self.current_route_traveldistance / current_route['length']) * (
                    current_route['elat'] - current_route['slat']) + current_route['slat']
                

                return False
                
        else:
            if not self.current_route['has_path']:
                # 如果没有路径，直接到达目的地
                distance = tbd.getdistance(
                    self.lon, self.lat, self.current_route['elon'], self.current_route['elat'])
                self.consume_power(distance)
                self.status = 0
                self.lon = self.current_route['elon']
                self.lat = self.current_route['elat']
            else:
                self.current_route_traveltime += self.step_length* config.CAR_SETTING['car_speed']
                  # 更新行驶时间
                # 需要进行行驶
                if self.position == {}:  # 还未在路径上
                    current_point = environment.find_path_position(
                        self.current_route, 0)  # 获取路径上的起点
                else:
                    current_point = self.position  # 获取当前位置
                target_point = environment.find_path_position(
                    self.current_route,
                    self.current_route_traveltime)
                # 计算行驶距离
                travel_distance = environment.get_length_between_points(
                    self.current_route, current_point, target_point)
                # 消耗电量
                self.consume_power(travel_distance)
                # 判断是否到达终点
                if target_point:  # 还未到达终点
                    self.position = target_point        # 更新在路径上的位置
                    position = environment.get_position(self.position)
                    self.lon = position[0]
                    self.lat = position[1]
                    return False
                else:   # 到达终点了
                    # 更新经纬度位置
                    self.lon = self.current_route['elon']
                    self.lat = self.current_route['elat']
                    # 清空在路网上的位置信息
                    self.position = {}

                    # 清空路径信息
                    self.current_route = {}
                    self.status = 0                     # 设置为停车状态
                    self.current_route_traveltime = 0
                    self.current_route_traveldistance = 0
                    return True
                
    def goto_location(self, environment, lon, lat):
        """
        执行车辆前往指定经纬度位置的任务。
        
        参数:
        - self: 指的是调用该方法的车辆对象实例。
        - environment: 指的是仿真环境对象，包含了仿真过程中的各种参数和状态信息。
        - lon: 目标位置的经度。
        - lat: 目标位置的纬度。
        
        返回值:
        - 无。该方法直接更新车辆的状态和路径信息，不返回值。
        
        详细说明:
        - 该方法首先调用仿真环境的find_travel_path函数，根据车辆当前的位置和目标位置计算出行的路径。
        - 计算得到的路径信息被更新到车辆对象的current_route属性中，以便后续的traveling_per_step方法使用。
        - 同时，车辆的current_route_traveldistance和current_route_traveltime属性被重置为0，表示开始新的行程。
        - 最后，车辆的状态被设置为1，即行驶状态，表示车辆已经开始前往目标位置的行程。
        """
        # 出行任务：去往某地
            
        # 获取路径
        route = environment.find_travel_path(self.lon, self.lat, lon, lat)
        # 更新路径信息
        self.current_route = route
        self.current_route_traveldistance = 0
        self.current_route_traveltime = 0
        # 设置为行驶状态
        self.status = 1

    def consume_power(self, distance):
        """
        根据给定的行驶距离计算并消耗车辆的电量。
        
        参数:
        - self: 指的是调用该方法的车辆对象实例。
        - distance: 需要计算消耗电量的行驶距离，单位为米。
        
        返回值:
        - 无。该方法直接更新车辆的电量属性，不返回值。
        
        详细说明:
        - 该方法首先根据车辆的能耗率(consumption_rate)和行驶距离来计算所需的能量消耗量(energy_cost)。
            - 能耗率通常以kWh/km为单位，因此距离需要转换为公里（除以1000），并且总能耗以千瓦时(kWh)计算，所以距离还需要除以100000（将米转换为公里，并考虑能耗率的单位）。
        - 接着，从车辆当前的电量(battery)中减去计算出的能量消耗量(energy_cost)，以模拟车辆在行驶过程中电量的减少。
        - 最后，确保车辆的电量不会低于0，即电量不足时不会继续消耗，这样可以避免电量透支的情况发生。
        """
        # 给定行驶距离，消耗电量
        energy_cost = distance * self.consumption_rate / 100000
        self.battery -= energy_cost
        # 电量不能低于0
        self.battery = max(self.battery, 0)

    
    def goto_nearest_charge_station(self, environment):
        """
        寻找最近的充电站并前往。
        
        参数:
        - self: 指的是调用该方法的车辆对象实例。
        - environment: 指的是仿真环境对象，包含了仿真过程中的各种参数和状态信息。
        
        返回值:
        - 如果成功找到并选择了一个充电站，则返回True。
        - 如果没有找到充电站，则返回False。
        
        详细说明:
        - 该方法首先定义了一个最大搜索距离maxdis，用于限制寻找充电站的范围。
        - 调用仿真环境的find_nearest_station函数，根据车辆当前的经纬度位置(self.lon, self.lat)，搜索一定数量(k=500)的最近充电站。
        - 搜索结果nearest_stations是一组距离车辆当前位置最近的充电站信息。
        - 接下来，调用车辆对象的select_station方法，从搜索到的充电站中选择一个充电站selected_station。
        - 如果存在可选的充电站，即is_exist为True，则获取该充电站的经纬度信息(stationlon, stationlat)。
        - 调用车辆对象的goto_location方法，前往选定的充电站位置。
        - 同时，更新车辆对象的target_charge_station属性，记录目标充电站的ID。
        - 更新车辆对象的target_lon和target_lat属性，记录目标充电站的经纬度。
        - 最后，返回True表示成功找到并选择了一个充电站。
        - 如果没有找到充电站，则直接返回False。
        """
        # 找最近的一些充电站，选择其中一个，然后前往

        maxdis = 20000
        # 找到最近的充电站
        nearest_stations = environment.find_nearest_station(
            self.lon, self.lat, k=60, maxdis=maxdis)
        while(len(nearest_stations)==0):
            maxdis += 2000
            nearest_stations = environment.find_nearest_station(
            self.lon, self.lat, k=60, maxdis=maxdis)
            
        is_exist, selected_station = self.select_station(environment, nearest_stations)
        if is_exist:
            stationlon, stationlat = environment.station_loc[selected_station]
            # 前往分配任务
            self.goto_location(environment, stationlon, stationlat)
            self.target_charge_station = selected_station
            self.target_lon = stationlon #----------
            self.target_lat = stationlat #----------
            return True 
        return False

    def get_travel_task(self, environment):
        """
            从行程链中获取出行任务，并确保车辆有足够的电量来执行新的任务。
            
            参数:
            - self: 指的是调用该方法的车辆对象实例。
            - environment: 指的是仿真环境对象，包含了仿真过程中的各种参数和状态信息。
            
            返回值:
            - 无。该方法直接更新车辆的状态和行程任务，不返回值。
            
            详细说明:
            - 该方法首先检查车辆的行程链(tripchain)是否还有未完成的任务，并且当前电量(battery)是否足够执行任务。
                - 如果行程链不为空且电量大于0，表示车辆有待执行的任务且有足够的电量。
            - 如果当前任务的时间(time)已经小于或等于当前的时间戳(timestamp)，表示该任务已经到达预定的出发时间。
                - 从行程链中取出下一个任务(next_trip)，并将其从行程链中移除。
            - 更新车辆对象的状态，包括判断是否是回家(wether_home)或去工作地点(wether_work)的任务。
            - 调用车辆对象的goto_location方法，根据任务信息前往目标位置。
            - 重置目标充电站(target_charge_station)为空字符串，表示当前没有特定的充电任务。
        """
        # 从行程链中获取出行任务并且保证没电不拥有新的出行动作
        if len(self.tripchain) > 0 and self.battery>0:
            if self.tripchain[0]['time'] <= environment.timestamp:
                # 获取下一个出行任务
                next_trip = self.tripchain.pop(0)
                
                # 更新部分状态
                self.wether_home = next_trip['wether_home']
                self.wether_work = next_trip['wether_work']
                # 分配出行任务
                self.goto_location(
                    environment, next_trip['lon'], next_trip['lat'])
                self.target_charge_station =''
    # endregion
    # region 车辆：充电行为
    def charging_decision(self, environment, start_charge=True):
        """
        根据车辆的电量和当前的电价情况，决定是否要开始充电或继续充电。
        
        参数:
        - self: 指的是调用该方法的车辆对象实例。
        - environment: 指的是仿真环境对象，包含了仿真过程中的各种参数和状态信息。
        - start_charge: 布尔值，True表示判断是否要开始充电，False表示判断是否要继续充电。
        
        返回值:
        - 如果需要充电，返回True。
        - 如果不需要充电，返回False。
        
        详细说明:
        - 该方法首先根据车辆的当前电量(battery)和总电量(total_power)来判断是否需要开始充电。
            - 如果电量低于总电量的10%，则需要开始充电。
            - 还可以根据车辆的充电决策电量(charge_decision_power)来判断，如果电量低于该值与总电量的百分比，则考虑充电。
        - 如果是判断是否要开始充电(start_charge为True)，则进一步根据当前时间(hour)和电价概率(off_peak_prob, mid_peak_prob, high_peak_prob)来决定是否充电。
            - 根据当前小时，判断当前是处于高峰、中峰还是平峰电价时段，并相应地调整充电意愿(prob)。
            - 如果电量已经接近总电量的90%，则不需要充电。
            - 使用概率分布函数，结合当前的充电意愿和随机数生成，来决定是否要充电。
        - 如果是判断是否要继续充电(start_charge为False)，则调用charging_stop_habit方法来决定是否结束充电。
        """
        # start_charge: True表示判断是否要开始充电，False表示判断是否要继续充电
        # 决定是否要充电或继续充电，返回True为充电，返回False为不充电
        if start_charge:
            if self.battery < self.total_power*0.1:
                return True
            if self.battery < (self.charge_decision_power*self.total_power)/100:
            
                hour = (int(environment.timestamp/60))%24    # 区分时间段
                prob = environment.off_peak_prob
                for h in environment.mid_peak_price_period:
                    if h == hour:
                        prob = environment.mid_peak_prob
                        break
                for h in environment.peak_price_period:
                    if h==hour:
                        prob = environment.high_peak_prob
                        break
                prob *= config.ENV_SETTING['charge_willings'][hour]
                if self.battery >= (0.9*self.total_power):
                        return False
                power_flag = 0
                if self.battery < self.total_power * 0.1:
                    return True
                ## 使用概率分布函数判断是否要充电
                if self.battery < (self.charge_decision_power*self.total_power)/100:
                    power_flag = power_flag + 1
                if power_flag > 0 :
                    random_num = random.random()
                    charge_flag = 0
                    if random_num <= prob:
                        charge_flag += 1

                    if charge_flag > 0 :
                        return True
                    else:
                        return False
            return False    
            
        else:
            # 判断是否需要结束充电
            return self.charging_stop_habit()

    # 充电停止行为
    def charging_stop_habit(self):
        """
        根据车辆当前电量与总电量的关系，决定是否停止充电。
        
        参数:
        - self: 指的是调用该方法的车辆对象实例。
        
        返回值:
        - 如果车辆需要停止充电，返回True。
        - 如果车辆不需要停止充电，返回False。
        
        详细说明:
        - 该方法首先检查车辆的当前电量(battery)是否已经达到或超过总电量(total_power)。
            - 如果当前电量等于或大于总电量，表示车辆已经充满电，此时应该停止充电。
            - 如果当前电量小于总电量，表示车辆尚未充满电，此时应该继续充电。
        - 基于上述逻辑，方法返回相应的布尔值，以指示是否应该停止充电。
        """
        if self.battery >= self.total_power:
            return False 
        else:
            return True
    
    def charging_per_step(self, station_agent):
        """
        模拟车辆在仿真过程中的每一步充电行为。
        
        参数:
        - self: 指的是调用该方法的车辆对象实例。
        - station_agent: 指的是充电站代理对象，包含了充电站的相关信息和状态。
        
        返回值:
        - 返回车辆当前的充电速度。
        
        详细说明:
        - 该方法首先将车辆的状态设置为充电状态，通过设置status属性为2来表示。
        - 记录当前充电站的ID，以便跟踪车辆的充电位置。
        - 计算当前的充电速度，取车辆的最大充电速度(charge_speed_car)和充电站的最大充电速度(charge_speed_station)中的较小值。
        - 根据当前充电速度和仿真步长(step_length)计算本步充电量(charge_power)，单位为千瓦时(kWh)。
        - 将计算出的充电量加到车辆的电量(battery)上，以模拟充电过程。
        - 确保车辆的电量不会超过电池的总容量(total_power)，以防止过充。
        - 最后，返回当前的充电速度，用于输出或进一步的逻辑处理。
        """
        # 进行充电
        self.status = 2  # 设置为充电状态
        self.charge_station = station_agent.station_id  # 设置充电站ID
        # 当前充电速度
        self.current_charge_speed = min(self.charge_speed_car,
                                        station_agent.charge_speed_station)
        
        # 当前步的充电量(kWh)
        charge_power = self.current_charge_speed * self.step_length / 60
        # 进行充电
        self.battery += charge_power
        # 充满
        self.battery = min(self.battery, self.total_power)
        
        # 输出当前充电速度
        return self.current_charge_speed

    def charging_start(self, environment, station_id):
        """
        模拟车辆在仿真过程中的每一步充电行为。
        
        参数:
        - self: 指的是调用该方法的车辆对象实例。
        - station_agent: 指的是充电站代理对象，包含了充电站的相关信息和状态。
        
        返回值:
        - 返回车辆当前的充电速度。
        
        详细说明:
        - 该方法首先将车辆的状态设置为充电状态，通过设置status属性为2来表示。
        - 记录当前充电站的ID，以便跟踪车辆的充电位置。
        - 计算当前的充电速度，取车辆的最大充电速度(charge_speed_car)和充电站的最大充电速度(charge_speed_station)中的较小值。
        - 根据当前充电速度和仿真步长(step_length)计算本步充电量(charge_power)，单位为千瓦时(kWh)。
        - 将计算出的充电量加到车辆的电量(battery)上，以模拟充电过程。
        - 确保车辆的电量不会超过电池的总容量(total_power)，以防止过充。
        - 最后，返回当前的充电速度，用于输出或进一步的逻辑处理。
        """
        # 开始在某个站点充电
        self.charge_station = station_id
        environment.agent_dict['station'][station_id].add_to_queue(self)
        

    def charging_stop(self, environment):
        """
        停止车辆的充电过程，并更新相关状态。
        
        参数:
        - self: 指的是调用该方法的车辆对象实例。
        - environment: 指的是仿真环境对象，包含了仿真过程中的各种参数和状态信息。
        
        返回值:
        - 无。该方法直接更新车辆的状态和充电站的队列信息，不返回值。
        
        详细说明:
        - 首先，将车辆的当前充电速度设置为0，表示停止充电。
        - 然后，检查车辆是否有正在充电的充电站ID，如果没有，则不需要执行停止充电的操作。
        - 如果车辆正在某个充电站充电，即存在于该充电站的current_queue队列中，则将其从队列中移除，并更新充电站的剩余充电车辆数量(num_charge_car_left)。
        - 如果车辆在充电站的waiting_queue等待队列中，同样将其移除。
        - 接着，将车辆的状态设置为0，表示车辆当前没有进行任何操作（非行驶、非充电状态）。
        - 最后，清空车辆的充电站ID，表示车辆已经离开充电站。
        - 通过这个方法，可以模拟车辆完成充电后离开充电站的过程，确保充电站资源的合理分配和车辆状态的准确记录。
        """
        self.current_charge_speed = 0 # 当前充电速度设置为0
        if self.charge_station != '':  # 如果有充电站
            # 停止充电
            if self in environment.agent_dict['station'][self.charge_station].current_queue:
                environment.agent_dict['station'][self.charge_station].current_queue.remove(self)
                environment.agent_dict['station'][self.charge_station].num_charge_car_left+=1
            elif self in environment.agent_dict['station'][self.charge_station].waiting_queue:
                environment.agent_dict['station'][self.charge_station].waiting_queue.remove(self)
            self.status = 0
            self.charge_station = ''

    def normalize(self, value, max_value, min_value):
        if max_value== min_value:
            return 1
        return (value-min_value)/(max_value-min_value)

    # 选择充电站
    def select_station(self, environment, stations):
        """
        选择一个合适的充电站进行充电。

        参数:
        self: 当前对象的引用，用于访问对象的属性和方法。
        environment: 环境对象，包含了环境的相关信息，如时间戳、代理字典等。
        stations: 一个列表，包含了所有可选的充电站的元组，每个元组包含充电站的ID和到当前位置的距离。

        返回:
        一个元组，第一个元素是一个布尔值，表示是否成功选择了充电站；第二个元素是选定的充电站的ID。
        """
        # 如果有私桩则选择私桩
        if ((self.has_private_pile == 1) & (self.wether_home==1)):
            return True, str(int(self.carid))+"_private"
        if config.CAR_SETTING['station_selection_method'] == 1:
            ## 计算得到hour
            hour = (int(environment.timestamp/60))%24
            max_price = 0
            min_price = 999999999
            max_distance = 0
            min_dictance = 99999999
            max_capacity = 0
            min_capacity = 999999999
            for station in stations:
                distance = station[1]
                capacity = environment.agent_dict['station'][station[0]].capacity()
                price = environment.agent_dict['station'][station[0]].unit_price[hour]
                max_distance = max_distance if max_distance > distance else distance
                min_dictance = min_dictance if min_dictance < distance else distance
                max_price = max_price if max_price > price else price 
                min_price = min_price if min_price < price else price
                max_capacity = max_capacity if max_capacity > capacity else capacity
                min_capacity = min_capacity if min_capacity < capacity else capacity
            score_list = []
            # 设定各个因素的权重
            weights = config.ENV_SETTING['preference_weights']
            for station in stations:
                distance = station[1]
                capacity = environment.agent_dict['station'][station[0]].capacity()
                price = environment.agent_dict['station'][station[0]].unit_price[hour]
                distance = self.normalize(distance,max_distance,min_dictance)
                capacity = self.normalize(capacity, max_capacity, min_capacity)
                price = self.normalize(price, max_price, min_price)
                score = (1-price)*weights['price'] + (1-distance)*weights['distance']+capacity*weights['capacity']
                score_list.append(score)
            score_sum = sum(score_list)
            score_list = [x/score_sum for x in score_list]
            select_station = random.choices(stations, weights=score_list,k=1)[0]
            return True, select_station[0]
                
        if config.CAR_SETTING['station_selection_method'] == 2:
            hour = (int(environment.timestamp/60))%24
            # 筛选出能够充电的充电站
            min_unit_price = 100000.00
            min_unit_price_station = None
            can_select_stations = []
            for station in stations:
                if environment.agent_dict['station'][station[0]].has_capacity():
                    can_select_stations.append((station[0],environment.agent_dict['station'][station[0]].unit_price))
                    if environment.agent_dict['station'][station[0]].unit_price[hour]<min_unit_price:
                        min_unit_price = environment.agent_dict['station'][station[0]].unit_price[hour]
                        min_unit_price_station = station[0]
            ## 进一步丰富充电站选择
            if len(can_select_stations)<=0:
                return False, min_unit_price_station
            return True, min_unit_price_station
        
        if config.CAR_SETTING['station_selection_method'] == 3:
            if len(stations) == 0:
                return False, -1
            station_capacity = []
            hour = (int(environment.timestamp/60))%2
            for station in stations:
                remaining_capacity = environment.agent_dict['station'][station[0]].remaining_capacity
                prob = 0
                if environment.weekday:
                    prob = environment.holiday_prob[environment.holiday_prob['stationId']==station[0]][str(hour)+'h'].iloc[0]
                else:
                    prob = environment.workday_prob[environment.workday_prob['stationId']==station[0]][str(hour)+'h'].iloc[0]
                # unit_price = environment.agent_dict['station'][station[0]].unit_price
                station_capacity.append([station[0],remaining_capacity,prob])

            # station_capacity = sorted(station_capacity,key=lambda x:x[2])
            has_capacity = [x for x in station_capacity if x[1]>0]
            sta = [x[0] for x in station_capacity]
            prob = [x[2]for x in station_capacity]
            if len(has_capacity) > 0:
                if np.array(prob).sum() == 0:
                    return True, random.choice(sta)
                return True, np.random.choice(sta,p=np.array(prob)/np.array(prob).sum())
            else:
                return True, np.random.choice(sta)
        
        
    
    def run(self, environment):
        """
        执行车辆在仿真环境中的主要运行逻辑，包括获取出行任务、行驶、充电等行为。
        
        参数:
        - self: 指的是调用该方法的车辆对象实例。
        - environment: 指的是仿真环境对象，包含了仿真过程中的各种参数和状态信息。
        
        返回值:
        - 无。该方法直接更新车辆的状态和行为，不返回值。
        
        详细说明:
        - 该方法是车辆对象在仿真环境中运行的核心方法，负责模拟车辆的完整行为过程。
        - 首先，初始化一系列成本变量，用于记录各个行为的时间消耗。
        - 检查车辆是否有当前行程路线，如果有，则设置车辆状态为行驶状态。
        - 调用get_travel_task方法获取车辆的出行任务，并记录时间消耗。
        - 如果车辆处于行驶状态，调用traveling_per_step方法进行行驶，并判断是否到达目的地，记录时间消耗。
        - 如果到达目的地，根据情况决定是否开始充电，并调用charging_start方法，记录时间消耗。
        - 如果车辆处于充电状态，调用charging_decision方法决定是否继续充电，如果决定停止充电，则调用charging_stop方法，记录时间消耗。
        - 如果车辆处于停车状态，根据充电决策决定是否前往充电站，并调用goto_nearest_charge_station方法，记录时间消耗。
        - 最后，调用save_infos方法保存车辆的信息至仿真环境，以供后续分析和处理。
        """

        time_start = time.time()##
        self.get_travel_task_cost = 0
        self.traveling_per_step_cost = 0
        self.charging_start_cost = 0
        self.find_nearest_station_cost = 0
        self.charging_decision_cost = 0
        self.select_station_cost = 0
        self.goto_charge_cost = 0

        if len(self.current_route) >0 :
            self.status = 1
        
        # 查看是否有车辆出行任务
        self.get_travel_task(environment)
        self.get_travel_task_cost = time.time()-time_start##

        #检查车辆运行决策
        if self.status == 1:  ## 行驶状态

            time_start = time.time()##

            is_end = self.traveling_per_step(environment)  # 行驶，并判断是否到达终点

            self.traveling_per_step_cost = time.time()-time_start##
            
            if is_end: #到达终点

                self.status = 0
                if self.target_charge_station:
                    time_start = time.time()##
                    # 如果有充电站则说明专门过来充电的
                    self.charging_start(environment, self.target_charge_station)
                    self.target_charge_station =''
                    self.charging_start_cost = time.time()-time_start##
                
        elif (self.status == 2) | (self.status == 3): #充电状态
           
            time_start = time.time()##

            if self.charging_decision(environment, start_charge=False): 
                #决定是否继续充电
                pass 
            else:
                #停止充电
                self.charging_stop(environment)

            self.charging_decision_cost = time.time()-time_start##

        elif self.status == 0: #停车状态
            if self.charging_decision(environment):

                time_start = time.time()##
                self.goto_nearest_charge_station(environment)
                self.goto_charge_cost = time.time()-time_start##
                    
            else:
                # 不需要充电， 仅停车
                pass
       
        self.save_infos(environment)
        #self.save_charge_infos(environment)