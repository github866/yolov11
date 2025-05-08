import json
import numpy as np


def load_polygons(file_name):
    with open(file_name, "r") as f:
        polygons_dict = json.load(f)
    # Keep both names and polygons
    room_names = list(polygons_dict.keys())
    polygons = list(polygons_dict.values())
    return polygons, room_names

def point_in_polygon(point, polygon):
    x, y = point
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def mask_location(polygons):
    # Create a 1920x1080 mask initialized with -1 (no polygon)
    mask = np.full((1080, 1920), -1, dtype=np.int32)
    
    # For each polygon, check every point in the mask
    for polygon_idx, polygon in enumerate(polygons):
        for y in range(1080):
            for x in range(1920):
                if point_in_polygon((x, y), polygon):
                    mask[y, x] = polygon_idx
    
    return mask

def main():
    polygons, room_names = load_polygons("room_segments/polygons.json")
    mask = mask_location(polygons)
    print("Mask created")

if __name__ == "__main__":
    main()

