import numpy as np

TYPES = [
    'Hoodie', 'Jacket', 'Mid-lenght dress', 'Pants', 'Shirt', 'coat', 'dress', 'fabric',
    'jacket', 'jean', 'm2m', 'plain', 'shirt', 'short', 'shorts', 'skirt',
    'slacks', 'suit', 'sweat', 'tie', 'tracksuit', 'tshirt'
    ]

COLORS = [
    "red", "green", "blue", "yellow", "cyan", "magenta", "black", "white", 
    "gray", "orange", "purple", "pink", "brown"
    ]

TYPE_TOPS = [
    "hoodie", "jacket", "shirt", "coat",
    "sweat", "tshirt", "m2m", "plain", "mid-lenght dress", "dress",
]

TYPE_BOTTOMS = [
    "pants", "jean", "short", "shorts",
    "skirt", "slacks", "tracksuit"
]

TYPE_OTHERS = [
    "fabric", "suit", "tie"
]

# Loop type and color and create a dictionary
TYPE_INDEX  = {t:i for i,t in enumerate(TYPES)}
COLOR_INDEX = {c:i for i,c in enumerate(COLORS)}

def embed(type_name, color_name):
    v_type  = np.zeros(len(TYPES))
    v_color = np.zeros(len(COLORS))

    # if type_name in TYPES and color_name in COLORS:
    if type_name in TYPE_INDEX:
        v_type[TYPE_INDEX[type_name]] = 1
    if color_name in COLOR_INDEX:
        v_color[COLOR_INDEX[color_name]] = 1

    # Create one long embedding vector
    v = np.concatenate([v_type, v_color])

    # Normalize the vector
    return v / (np.linalg.norm(v) + 1e-10)

# Measure cosine similarity
def cos(a, b):
    return float(np.dot(a,b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-10))

def split_item(item_list):
    items = [t.split(" ") for t in item_list]

    TOPS = []
    BOTTOMS = []
    
    for t,c in items:
        if t in TYPE_TOPS:
            TOPS.append((t,c))
        elif t in TYPE_BOTTOMS:
            BOTTOMS.append((t,c))
        else:
            continue

    return [TOPS, BOTTOMS]

def recommend(item_type, item_color, item_list):
    TOPS, BOTTOMS = split_item(item_list);
    query = embed(item_type, item_color)

    if item_type in [t for t,c in TOPS]:
        # user picked TOP → recommend BOTTOM
        candidates = BOTTOMS
    else:
        # user picked BOTTOM → recommend TOP
        candidates = TOPS

    scored = []
    if(candidates == []):
       return ["No matching items found"]

    for t,c in candidates:
        score = cos(query, embed(t,c))
        scored.append(((t,c), score))

    return sorted(scored, key=lambda x:x[1], reverse=True)