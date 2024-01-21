import sys
import math
from collections import defaultdict

from pdb import set_trace

def read_city_gps():
    city_gps_data = {}
    with open('city-gps.txt', 'r') as file:
        for line in file:
            fields = line.strip().split()
            city = fields[0]
            latitude = float(fields[1])
            longitude = float(fields[2])
            city_gps_data[city] = (latitude, longitude)
    return city_gps_data

def read_road_segments():
    road_optimum_segments = {}
    with open('road-segments.txt', 'r') as file:
        for line in file:
            fields = line.strip().split()
            city1 = fields[0]
            city2 = fields[1]
            length = float(fields[2])
            speed_limit = int(fields[3])
            highway_nm = fields[4]
            road_optimum_segments[(city1, city2)] = (length, speed_limit, highway_nm)
            road_optimum_segments[(city2, city1)] = (length, speed_limit, highway_nm)
    return road_optimum_segments

def cost_delivery_time(length, speed_limit, trip_t):
    road_t = length/speed_limit
    if speed_limit>=50:
        probability = math.tanh(length / 1000)
        return road_t + ( probability * 2 * (road_t + trip_t) )
    else:
        return road_t

def cal_travel_time(length, speed_limit):
    return length / speed_limit

def cal_total(road_optimum_segments, final_routes):
    total_delivery_time = 0
    total_miles = 0
    total_time = 0
    return_route = []

    for route in final_routes:

        city_prev, city_current = route
        length, speed_limit, highway_nm = road_optimum_segments.get((city_prev, city_current), (0, 0, ""))
        
        road_t = length/speed_limit
        if speed_limit >= 50:
            probability = math.tanh(length / 1000)
            t_total = road_t + ( probability * 2 * (road_t + total_delivery_time) )
            total_delivery_time += t_total
            
        else:
            total_delivery_time += road_t

        total_miles += length
        total_time += cal_travel_time(length, speed_limit)
        segment_info = f"{highway_nm} for {length} miles at {speed_limit} mph"
        return_route.append((city_current, segment_info))

    return total_miles, total_time, total_delivery_time, return_route

def cal_distance(lat1, lon1, lat2, lon2):
    radius = 3959
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return radius*c

def get_route(start, end, cost_function):
    city_gps_data = read_city_gps()
    road_optimum_segments = read_road_segments()
    def heuristic(city1, city2, cost_function,func_val):
        city2neighbours = defaultdict(list)

        lat2, lon2 = city_gps_data.get(city2, (0, 0))
        glat1, glon2 = city_gps_data.get(end, (0,0))

        if (lat2, lon2) == (0, 0):
            return 0
        
        if cost_function == "segments":
            if city1 == city2:
                return 0
            return 1
        elif cost_function == "time":
            distance = cal_distance(lat2, lon2, glat1, glon2)
            return distance / speed_limit                                             
        elif cost_function == "distance":
            return cal_distance(lat2, lon2, glat1, glon2)
        elif cost_function == "delivery":


            l = cal_distance(lat2, lon2, glat1, glon2)
            
            p = math.tanh(l / 1000) 
            road_t = l / 50
            trip_t = func_val[city_current]
            return road_t + ( p * 2 * (road_t + trip_t) )

        
        else:
            raise ValueError( "cost function not provided")

    open_list = [(0, start)]
    visited = set()
    func_val = {city: float('inf') for city in city_gps_data}
    func_val[start] = 0
    parent = {}

    total_delivery_time = 0  # total delivery time initialization

    while open_list:
        _, city_current = min(open_list)
        open_list.remove((_, city_current))

        if city_current == end:

            # route from start to end reconstruction
            final_routes = []
            while city_current != start:
                city_prev = parent.get(city_current)
                final_routes.append((city_prev, city_current))
                city_current = city_prev
            final_routes.reverse()
            total_miles, total_time, total_delivery_time, return_route = cal_total(road_optimum_segments, final_routes)  

            return {"total-segments" : len(return_route), 
                    "total-miles" : total_miles,
                    "total-hours" : total_time, 
                    "total-delivery-hours" : total_delivery_time,  # Include delivery time here
                    "route-taken" :  return_route}

        visited.add(city_current)

        for neighbour, (length, speed_limit, _) in road_optimum_segments.items():
            if city_current == neighbour[0] and neighbour[1] not in visited:

                if cost_function == 'segments':
                    optimum_func = func_val[city_current] + 1
                if cost_function == 'distance':
                    optimum_func = func_val[city_current] + length
                if cost_function == 'time':
                    optimum_func = func_val[city_current] + cal_travel_time(length,speed_limit)
                if cost_function == 'delivery':
                    trip_t = func_val[city_current]
                    optimum_func = func_val[city_current] + cost_delivery_time(length, speed_limit, trip_t)

                if neighbour[1] not in func_val or optimum_func < func_val[neighbour[1]]:
                    func_val[neighbour[1]] = optimum_func
                    parent[neighbour[1]] = city_current
                    open_list.append((func_val[neighbour[1]] + heuristic(neighbour[1], end, cost_function, func_val), neighbour[1]))


    return None



# Please don't modify anything below this line

if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise Exception("Error: expected 3 arguments")

    (_, start_city, end_city, cost_function) = sys.argv
    if cost_function not in ("segments", "distance", "time", "delivery"):
        raise Exception("Error: invalid cost function")
    result = get_route(start_city, end_city, cost_function)

    print("Start in %s" % start_city)
    for step in result["route-taken"]:
        print("   Then go to %s via %s" % step)

    print("\n          Total segments: %4d" % result["total-segments"])
    print("             Total miles: %8.3f" % result["total-miles"])
    print("             Total hours: %8.3f" % result["total-hours"])
    print("Total hours for delivery: %8.3f" % result["total-delivery-hours"])

