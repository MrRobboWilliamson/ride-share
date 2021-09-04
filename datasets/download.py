import overpy
import pandas as pd

api = overpy.Overpass()

# get all streets in manhattan
result = api.query("""
    area[name="Manhattan"];
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