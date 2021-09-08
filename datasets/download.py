import overpy
import pandas as pd

api = overpy.Overpass()

# find the manhattan area
# result = api.query("""
#     is_in(40.766052, -73.968787)->.a;
# rel(pivot.a)[boundary=administrative];
# out tags;
#     """)

# for foo in result.relations:
#     print(foo.id,foo.tags.get("name"))

# get all streets in manhattan
manhattan_id = 8398124
result = api.query(f"""
    rel({manhattan_id});map_to_area;
way[highway~"^(primary|secondary|tertiary|residential|unclassified|living_street)$"](area);
    (._;>;);
    out body;
    """)

data = []
types = set()
for way in result.ways:
    
    # get the street details
    sid = way.id
    name = way.tags.get("name", "n/a")
    type_ = way.tags.get("highway", "n/a")
    oneway = way.tags.get("oneway", "n/a")
    
    # get the nodes
    for node in way.nodes:
        
        # get the node details
        nid = node.id
        lat = node.lat
        lon = node.lon
        
        record = [sid,name,type_,oneway,nid,lat,lon]
        data.append(record)
        

df = pd.DataFrame(data,columns=['street','name','type','oneway','node','lat','lon'])
df.to_csv("nyc_streets.csv",index=False)
