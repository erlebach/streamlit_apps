from node import Node
from walker import Walker

a = Node(ID = 'A')
b = Node(ID = 'B')
c = Node(ID = 'C')
d = Node(ID = 'D')
e = Node(ID = 'E')
f = Node(ID = 'F')
g = Node(ID = 'G')
h = Node(ID = 'H')
i = Node(ID = 'I')
j = Node(ID = 'J')
k = Node(ID = 'K')
l = Node(ID = 'L')
m = Node(ID = 'M')
n = Node(ID = 'N')
o = Node(ID = 'O')

# level 0
o.children = [e, f, n]

# level 1
e.parent = o
e.right_sibling = f
e.children = [a, d]

f.parent = o
f.left_sibling = e
f.right_sibling = n

n.parent = o
n.left_sibling = f
n.children = [g, m]

# level 2
a.parent = e
a.right_sibling = d

d.parent = e
d.left_sibling = a
d.children = [b, c]

g.parent = n
g.right_sibling = m

m.parent = n
m.left_sibling = g
m.children = [h, i, j, k, l]

# level 3
b.parent = d
b.right_sibling = c

c.parent = d
c.left_sibling = b

h.parent = m
h.right_sibling = i

i.parent = m
i.left_sibling = h
i.right_sibling = j

j.parent = m
j.left_sibling = i
j.right_sibling = k

k.parent = m
k.left_sibling = j
k.right_sibling = l

l.parent = m
l.left_sibling = k

w = Walker(debug=True, rootX = 0, rootY = 0)
w.config['NODE_SIZE'] = 2
w.config['NODE_SEPARATION'] = 4
w.config['TREE_SEPARATION'] = 4

w.add_nodes([o, e, f, n, a, d, g, m, b, c, h, i, j, k, l])

w.position_tree()
