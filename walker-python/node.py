from __future__ import annotations

class Node:
  """
  #### Node object to be used in Walker's algorithm.

  Properties:
  - `parent`: type `Node | None`, parent of the node
  - `left_sibling`: type `Node | None`, left sibling of the node
  - `right_sibling`: type `Node | None`, right sibling of the node
  - `children`: type `list[Node]`, children of the node
  - `id`: type `int`, ID of the node
  - `x`, `y`: type `int`, coordinates of the node
  - `prelim`: type `int`, preliminary x coordinates of the node
  - `modifier`: type `int`, modifier value of the node
  - `left_neighbor`: type `Node`, nearest neighbor of the node to the left at the same level

  Methods:
  - `__init__(parent = None, lSib = None, rSib = None, children = None, id = 0)`: constructor, returns `None`
  - `is_leaf()`: check whether the node is a leaf, returns `bool`
  - `is_leftmost()`: check whether the node is the leftmost child of its parent, returns `bool`
  - `is_rightmost()`: check whether the node is the rightmost child of its parent, returns `bool`
  - `get_leftmost_child()`: get the leftmost child of the node, returns `None` if the node is a leaf, otherwise `Node`
  - `get_rightmost_child()`: get the rightmost child of the node, returns `None` if the node is a leaf, otherwise `Node`
  - `get_leftmost(curr_level, search_depth)`: get the leftmost descendant of the node at a given depth, returns `None` if the node is a leaf, otherwise `Node`

  """
  def __init__(self, parent: Node or None = None,
               lSib: Node or None = None,
               rSib: Node or None = None,
               children: list[Node] or None = None,
               ID: int or str = 0) -> None:
    """
    constructor

    Parameters:
    - `parent`: type `Node | None`, optional, parent of the new node if exists, defaults to `None`
    - `lSib`: type `Node | None`, optional, left sibling of the new node if exists, defaults to `None`
    - `rSib`: type `Node | None`, optional, right sibling of the new node if exists, defaults to `None`
    - `children`: type `list[Node]`, optional, list of children of the new node if exists, defaults to `[]`
    - `id`: type `int | str`, ID of the new node, defaults to `0`

    Returns: `None`
    """
    self.parent = parent
    self.left_sibling = lSib
    self.right_sibling = rSib
    self.children = children if children is not None else []
    self.id = ID
    self.x = 0
    self.y = 0
    self.prelim = 0
    self.modifier = 0
    self.left_neighbor: Node or None = None

  def is_leaf(self) -> bool:
    """
    check whether the node is a leaf

    Parameters: None

    Returns: `bool`
    """
    return len(self.children) == 0

  def is_leftmost(self) -> bool:
    """
    check whether the node is the leftmost child of its parent

    Parameters: None

    Returns: `bool`
    """
    if self.parent is None: return True
    return self.parent.children[0] is self

  def is_rightmost(self) -> bool:
    """
    check whether the node is the rightmost child of its parent

    Parameters: None

    Returns: `bool`
    """
    if self.parent is None: return True
    return self.parent.children[-1] is self

  def get_leftmost_child(self) -> Node or None:
    """
    get the leftmost child of the node

    Parameters: None

    Returns: `None` if the node is a leaf, otherwise `Node`
    """
    if len(self.children) == 0: return None
    return self.children[0]

  def get_rightmost_child(self) -> Node or None:
    """
    get the rightmost child of the node

    Parameters: None

    Returns: `None` if the node is a leaf, otherwise `Node`
    """
    if len(self.children) == 0: return None
    return self.children[-1]

  def get_leftmost(self, curr_level: int, search_depth: int) -> Node or None:
    """
    get the leftmost descendant of the node at a given depth

    Parameters:
    - `curr_level`: type `int`, required, current level of search
    - `search_depth`: type `int`, required, the target level of search

    Returns: `None` if the node is a leaf, otherwise `Node`
    """
    if curr_level >= search_depth:
      return self
    elif self.is_leaf():
      return None
    else:
      curr_rightmost = self.get_leftmost_child()
      curr_leftmost = curr_rightmost.get_leftmost(curr_level + 1, search_depth)

      while curr_leftmost is None and curr_rightmost.right_sibling is not None:
        curr_rightmost = curr_rightmost.right_sibling
        curr_leftmost = curr_rightmost.get_leftmost(curr_level + 1, search_depth)

      return curr_leftmost
