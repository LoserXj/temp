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
import pickle
from datetime import timedelta
gc.disable()



class EnvironmentAgent:
    def __init__(
            self, use_road_network,
            agent_dict ,
            step_length,
            start_time,
            workday_prob,
            holiday_prob,
            route_cache={},
            save_car_park_infos=False):
        self.direct = not use_road_network
        self.agent_dict = agent_dict    # agent对象
        self.all_agents = []
        self.car_loc = {}               # 车辆所在位置
        self.station_loc = {}           # 站点所在位置
        self.history_car_loc = []       # 车辆历史位置
        self.timestamp = 0              # 当前时间
        self.ev_propotion = 0.1         # 仿真汽车比例（用于扩样流量，反推车速）
        self.step_length = step_length
        self.update_station_pos()       # 更新充电站位置
        self.infos = []                 # 用于存储每一仿真步的信息
        self.step = -1                   # 仿真步数
        self.route_cache = route_cache                      # 路网缓存
        self.save_car_park_infos = save_car_park_infos    # 是否保存停车时的记录
        self.off_peak_price_period = config.ENV_SETTING['off_peak_price_period']
        self.mid_peak_price_period = config.ENV_SETTING['mid_peak_price_period']
        self.peak_price_period = config.ENV_SETTING['peak_price_period']
        self.off_peak_prob = config.ENV_SETTING['off_peak_prob']
        self.mid_peak_prob = config.ENV_SETTING['mid_peak_prob']
        self.high_peak_prob = config.ENV_SETTING['high_peak_prob']
        self.start_time = start_time
        self.current_time = self.start_time + timedelta(minutes=0)
        self.weekday = self.is_weekday()
        self.workday_prob = workday_prob 
        self.holiday_prob = holiday_prob
        self.init_road_cache()
    
    def is_weekday(self):
        return self.current_time.weekday()>4

    def set_road_network(
        self,
            G,
        hwy_speeds={"residential": 10, "secondary": 15,
                    'primary': 15, "tertiary": 20}
        # 设置路网
    ):
        """
        设置仿真环境的路网，并计算每条边的出行时长。
        
        参数:
        - self: 指的是调用该方法的对象实例，通常为仿真环境类的一个实例。
        - G: 指的是一个networkx的MultiDiGraph对象，代表路网图。
        - highway_speeds: 一个字典，包含了不同类型道路的默认速度。
        
        返回值:
        - 无。该方法直接更新调用对象的road_network属性和road_network_edge属性，不返回值。
        
        详细说明:
        - 该方法接收一个networkx的MultiDiGraph对象G作为输入，该对象代表了一个路网图。
        - 首先，使用ox.graph_to_gdfs函数将路网图G转换为GeoDataFrame格式的节点和边数据。
        - 然后，对转换后的边数据gdf_edges进行处理，重置索引并删除重复的边，确保每对节点之间只保留一条边。
        - 重新构建路网图G，使用gdf_nodes和gdf_edges作为节点和边的数据源，并保留原始图G的图属性。
        - 通过ox.utils_graph.get_largest_component函数获取路网的最大强连通子图。
        - 使用ox.add_edge_speeds和ox.add_edge_travel_times函数计算每条边的出行时长，并更新到路网图G的边属性中。
        - 最后，将处理后的路网图G和边数据保存到调用对象的road_network和road_network_edge属性中，供后续仿真使用。
        """
        # 设置路网
        gdf_nodes, gdf_edges = ox.graph_to_gdfs(G)  # 将路网转换为GeoDataFrame
        # 连接两个节点只保留一条边
        gdf_edges = gdf_edges.reset_index().drop_duplicates(
            subset=['u', 'v'], keep='first')
        gdf_edges['key'] = 0
        gdf_edges = gdf_edges.set_index(['u', 'v', 'key'])

        # 重新构建路网
        # 图属性,这里使用了之前的路网数据的图属性
        G = ox.graph_from_gdfs(gdf_nodes, gdf_edges, graph_attrs=G.graph)
        # 路网最大强连通子图
        G = ox.utils_graph.get_largest_component(G,
                                                 strongly=True  # 是否强连通
                                                 )
        # 计算每条边出行时长
        G = ox.add_edge_speeds(G, hwy_speeds)
        G = ox.add_edge_travel_times(G)

        self.road_network = G
        self.road_network_edge = ox.utils_graph.graph_to_gdfs(
            G, nodes=False, fill_edge_geometry=False)

    def init_road_cache(self):
        """
        初始化路网缓存，加载预先计算的路径和路网图。
        
        详细说明:
        - 该方法首先检查是否需要使用路网(SIM_SETTING['use_road_network'])。
        - 如果需要，从文件中加载路网图(G)和两个缓存文件(route_cache_1和route_cache_2)。
        - 将两个缓存文件中的路径信息合并，并更新到调用对象的route_cache属性中。
        - 调用set_road_network方法设置路网图G，并计算每条边的出行时长。
        """
        if config.SIM_SETTING['use_road_network']:
            G = ox.load_graphml(config.CACHE_SETTING['shanghai_road'])
            with open(config.CACHE_SETTING['cache_road_path'],'rb') as file:
                route_cache_1 = pickle.load(file)
            with open(config.CACHE_SETTING['cache_road_path_runtime'],'rb') as file:
                route_cache_2 = pickle.load(file)
            route_cache_1.update(route_cache_2)
            self.route_cache = route_cache_1
            self.set_road_network(G)

    # 计算u，v的hash值
    def generate_unique_key(self, slon, slat, elon, elat):
        """
        计算给定坐标的哈希值，用于路径缓存的键。
        
        参数:
        - slon, slat: 起点的经度和纬度。
        - elon, elat: 终点的经度和纬度。
        
        返回值:
        - 返回一个字符串，代表坐标的MD5哈希值。
        """
        coordinates_str = f"{round(slon, 4)},{round(slat, 4)},{round(elon, 4)},{round(elat, 4)}"
        return hashlib.md5(coordinates_str.encode('utf-8')).hexdigest()

    # 只考虑直线路径
    def find_travel_path(self, lon1, lat1, lon2, lat2, shortest=1, weight='travel_time'):
        """
        根据给定的起止点坐标，寻找最短或随机路径。
        
        参数:
        - lon1, lat1: 起点的经度和纬度。
        - lon2, lat2: 终点的经度和纬度。
        - shortest: 指定寻找最短路径还是前k条最短路径中的随机一条，值为1时表示最短路径，大于1时表示前k条中的随机一条。
        - weight: 路径计算时考虑的权重，通常是'travel_time'，表示旅行时间。
        
        返回值:
        - 返回一个字典，包含路径信息，如路径是否直接、长度、旅行时间等。
        """
        # 引入 direct 为 True 时，只考虑直线路径
        if self.direct:
            travel_length = tbd.getdistance(
                lon1, lat1, lon2, lat2) * 1.5  # 道路非直线系数系数
            route = {
                'direct': True,
                'length': travel_length,
                'slon': lon1,
                'slat': lat1,
                'elon': lon2,
                'elat': lat2}
            return route
        else:
            # 计算uv的key值
            key_str = f"{round(lon1, 3)},{round(lat1, 3)},{round(lon2, 3)},{round(lat2, 3)}"
            if key_str in self.route_cache:
                route = self.route_cache[key_str].copy()
                route['slon'] = lon1
                route['slat'] = lat1
                route['elon'] = lon2
                route['elat'] = lat2
                return route
            # 找到两点间的最短出行路径
            # 由给定的坐标获取最近节点
            orig = ox.distance.nearest_nodes(self.road_network, X=lon1, Y=lat1)
            dest = ox.distance.nearest_nodes(self.road_network, X=lon2, Y=lat2)
            if shortest == 1:
                # 找到最短路径
                travel_route = ox.shortest_path(
                    self.road_network, orig, dest, weight=weight)
            if shortest > 1:
                # 前k最短路径中选择一个
                routes = ox.k_shortest_paths(
                    self.road_network, orig, dest, k=shortest, weight=weight)
                routes = list(routes)
                travel_route = routes[np.random.choice(range(len(routes)))]
            # 获取路径上的行驶时间
            travel_time = ox.utils_graph.get_route_edge_attributes(
                self.road_network, travel_route, attribute='travel_time')
            length = ox.utils_graph.get_route_edge_attributes(
                self.road_network, travel_route, attribute='length')
            # 将路径和行驶时间组合成字典
            route = {
                'direct': False,
                'travel_route': travel_route,
                'travel_time': travel_time,
                'length': length,
                'has_path': len(travel_route) > 1}
            self.route_cache[key_str] = route
            route_copy = route.copy()
            route_copy['slon'] = lon1
            route_copy['slat'] = lat1
            route_copy['elon'] = lon2
            route_copy['elat'] = lat2
            return route_copy
        

    def find_path_position(self, route, traveled_time):
        """
        根据已行驶的时间，在给定路径中找到当前位置。
        
        参数:
        - route: 一个字典，包含路径信息，其中'travel_time'键对应一个列表，包含路径上每一段的旅行时间；'travel_route'键对应一个列表，包含路径上的节点ID。
        - traveled_time: 一个整数或浮点数，表示已经行驶的时间（单位与'travel_time'中的单位相同）。
        
        返回值:
        - 返回一个字典，包含当前位置的详细信息：
          - 'u': 当前所在边的起点节点ID。
          - 'v': 当前所在边的终点节点ID。
          - 'index': 当前所在边在路径中的索引位置。
          - 'percentage': 在当前边中已行驶的百分比（0到1之间）。
        - 如果已行驶时间超过路径的总旅行时间，则返回None。
        """
        # 输入路径和已经行驶的时间，返回当前位置
        if traveled_time > sum(route['travel_time']):
            return None
        travel_time = route['travel_time']
        cumulative_sum = 0
        index = 0
        for i, time in enumerate(travel_time):
            cumulative_sum += time
            if cumulative_sum >= traveled_time:
                index = i
                traveled_time -= cumulative_sum - time
                break
        # 计算当前位置，占这条边长度的百分比
        if route['travel_time'][index] == 0:
            traveled_percentage = 0
        else:
            traveled_percentage = traveled_time / route['travel_time'][index]
        # 整理结果
        position = {'u': route['travel_route'][index],
                    'v': route['travel_route'][index+1],
                    'index': index,
                    'percentage': traveled_percentage}
        return position

    def get_length_between_points(self, route, position1, position2):
        """
        计算给定路径上两点之间的出行长度。
        
        参数:
        - route: 一个字典，包含路径信息，其中'length'键对应一个列表，包含路径上每一段的长度。
        - position1: 一个字典，包含路径上第一个位置的信息。
        - position2: 一个字典，包含路径上第二个位置的信息。如果未提供，则计算到路径的最后一个位置。
        
        返回值:
        - 返回两点之间的出行长度，单位与'length'中的单位相同。
        """
        # 获取两点之间的出行长度
        if not position2:
            position2 = self.find_path_position(
                route, sum(route['travel_time']))

        if position1['index'] == position2['index']:
            travel_length = route['length'][position1['index']] * \
                abs(position1['percentage']-position2['percentage'])
        elif position1['index'] < position2['index']:
            pathes = route['length'][position1['index']:position2['index']+1]
            travel_length = sum([pathes[0]*(1-position1['percentage'])] +
                                pathes[1:-1]+[pathes[-1]*position2['percentage']])
        else:
            travel_length = 0
        return travel_length

    def get_position(self, position):
        """
        根据位置信息，返回对应坐标。
        
        参数:
        - position: 一个字典，包含位置信息，应包含：
          - 'u': 边的起点节点ID。
          - 'v': 边的终点节点ID。
          - 'index': 边在路径中的索引位置。
          - 'percentage': 在边中的位置百分比（0到1之间）。
        
        返回值:
        - 返回一个坐标元组（经度，纬度），表示路径上该位置的地理坐标。
        """
        # 输入单个位置信息，返回位置的坐标
        # positions:[{'u':xx,'v':xx,'index':xxx,'percentage':xx},...]
        edges_geometry = self.road_network_edge.loc[(
            position['u'], position['v'], 0)]['geometry']

        point = edges_geometry.interpolate(
            position['percentage'], normalized=True)
        return point.coords[0]

    def find_nearest_station(self, lon, lat, k=1, maxdis=20000):
        """
        找到距离指定位置最近的充电站

        输入:
        lon (float): 经度
        lat (float): 纬度
        k (int): 要查找的最近充电站的数量，默认为1
        maxdis (float): 允许的最大距离，默认为1000

        输出:
        results (list): 包含最近充电站信息的列表，每个元素是一个包含充电站名称和距离的列表

        参数:
        - lon (float): 指定位置的经度
        - lat (float): 指定位置的纬度
        - k (int): 要查找的最近充电站的数量
        - maxdis (float): 允许的最大距离

        说明:
        此函数接受一个经度和纬度，并查找距离该位置最近的充电站。可以通过参数k指定要查找的最近充电站的数量，通过参数maxdis限制最大距离。
        返回包含最近充电站信息的列表，每个元素包含充电站名称和距离。
        """
        if self.station_KDtree == None:
            return []
        if k > len(self.station_loc):
            k = len(self.station_loc)
        # 执行最近邻查询
        _, nearest_idx = self.station_KDtree.query((lon, lat), k=k)
        if k == 1:
            nearest_idx = np.array([nearest_idx])
        # 通过最近邻索引找到对应的站点名称
        nearest_station_names = np.array(
            list(self.station_loc.keys()))[nearest_idx]
        # 计算充电站的距离
        results = [[nearest_station_name, tbd.getdistance(
            lon, lat, self.station_loc[nearest_station_name][0], self.station_loc[nearest_station_name][1])] for nearest_station_name in nearest_station_names]
        # 过滤掉距离大于maxdis的充电站
        results = [result for result in results if result[1] <= maxdis]
        return results

    def update_station_pos(self):
        """
        更新所有充电站的位置信息，并构建充电站的KD树以优化位置检索。

        详细说明:
        - 遍历仿真环境中所有的充电站代理，更新它们的地理位置信息。
        - 对于非私有充电站，将它们的位置信息存储到public_station_loc字典中。
        - 如果存在公共充电站，则使用这些位置信息构建一个KD树，以便后续快速检索最近的充电站。
        - 如果没有公共充电站，则将station_KDtree设置为None。
        """
        public_station_loc = dict()
        pattern = re.compile(r'_private$')
        # 更新所有充电站位置
        for _, station_agent in self.agent_dict['station'].items():
            self.station_loc[station_agent.station_id] = (
                station_agent.lon, station_agent.lat)
            if not pattern.search(station_agent.station_id):
                public_station_loc[station_agent.station_id] = (
                    station_agent.lon, station_agent.lat
                )
        # 构建充电站KD树以便后续检索
        if len(public_station_loc)!= 0:
            self.station_KDtree = KDTree(list(public_station_loc.values()))
        else:
            self.station_KDtree = None

    def save_infos(self):
        """
        保存仿真运行的相关信息。

        详细说明:
        - 将当前的仿真运行成本信息添加到infos列表中，包括车辆运行成本、充电站运行成本、时间戳和步骤数。
        """
        self.infos.append({
            'car_running_cost': self.car_running_cost,
            'station_running_cost': self.station_running_cost,
            'timestamp': self.timestamp,
            'step': self.step,
        })
        pass

    def run(self):
        """
        执行仿真环境中所有代理的运行逻辑。

        详细说明:
        - 首先增加仿真的步骤计数器，并更新当前时间戳。
        - 然后运行所有车辆代理的运行逻辑，并计算车辆运行的总成本。
        - 接着运行所有充电站代理的运行逻辑，并计算充电站运行的总成本。
        - 最后运行所有其他类型的代理，并保存当前步骤的仿真信息。
        """
        self.step += 1
        # 仿真运行
        self.timestamp = self.step * self.step_length  # 更新当前时间

        self.current_time = self.start_time + timedelta(minutes=self.timestamp)
        self.weekday = self.is_weekday()
        # 车辆运行
        time_start = time.time()
        [car_agent.run(self) for _, car_agent in self.agent_dict['car'].items()]
        self.car_running_cost = time.time()-time_start

        # 充电站运行
        time_start = time.time()
        [station_agent.run(self) for _, station_agent in self.agent_dict['station'].items()]
        self.station_running_cost = time.time()-time_start

        # 运行所有其他agent
        [agent.run(self) for agent in self.all_agents]

        self.save_infos()  # 保存信息

    def simulate(self, steps):
        """
        执行整个仿真过程，运行指定的步数。

        参数:
        - steps: 一个整数，表示仿真的总步数。

        详细说明:
        - 该方法首先打印出模型的基本信息，包括仿真步长、总步数、仿真天数、车辆数量和充电站数量。
        - 根据是否采用路网，打印出相应的信息。
        - 然后，使用一个for循环运行仿真steps次，每次循环代表一个仿真步。
        - 在每次仿真步中，调用run方法来执行仿真环境的所有代理的运行逻辑。
        - 如果配置中指定使用路网，仿真结束后将路径缓存保存到文件中，以便将来的仿真使用。
        """
        print('========================模型信息========================')
        print(f'仿真步长：{self.step_length}分钟')
        print(f'仿真步数：{steps}步')
        print(f'仿真天数：{int(steps*self.step_length/(60*24))}天')
        print('车辆数量：{}辆'.format(len(self.agent_dict['car'])))
        print('充电站数量：{}'.format(len(self.agent_dict['station'])))
        print(f'是否采用路网：{not self.direct}')
        print('========================开始仿真========================')
        for t in tqdm(range(steps), desc='正在仿真：'):
            self.run()  # 仿真运行
        if config.SIM_SETTING['use_road_network']:
            with open(config.CACHE_SETTING['cache_road_path_runtime'], 'wb') as f:
                pickle.dump(self.route_cache, f)
