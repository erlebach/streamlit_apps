from walker import Walker

DATA = './test.json'

w = Walker(debug=True, rootX = 0, rootY = 0)
w.config['NODE_SIZE'] = 2
w.config['NODE_SEPARATION'] = 4
w.config['TREE_SEPARATION'] = 4

w.add_nodes_from_json(DATA)
w.position_tree()
w.export_to_json('./out.json')
