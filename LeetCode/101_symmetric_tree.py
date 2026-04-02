'''
Given the root of a binary tree, check whether it is a mirror of itself (i.e., symmetric around its center).

 

Example 1:


Input: root = [1,2,2,3,4,4,3]
Output: true
Example 2:


Input: root = [1,2,2,null,3,null,3]
Output: false
 

Constraints:

The number of nodes in the tree is in the range [1, 1000].
-100 <= Node.val <= 100
'''

# solution
from typing import Optional

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

# solution (recursive)
class Solution:
    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        if not root:
            return True
        
        return self.isMirror(root.left, root.right)

    def isMirror(self, t1: Optional[TreeNode], t2: Optional[TreeNode]) -> bool:
        if not t1 and not t2:
            return True
        if not t1 or not t2:
            return False

        return (t1.val == t2.val) and \
               self.isMirror(t1.left, t2.right) and \
               self.isMirror(t1.right, t2.left)
    
# solution (iterative)
class Solution:
    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        if not root :
            return True

        queue = []

        queue.append([root.left, root.right])

        while queue :
            root_left, root_right = queue.pop(0)

            if not root_left and not root_right :
                continue
            
            if not root_left or not root_right or root_left.val != root_right.val :
                return False

            queue.append([root_left.left, root_right.right])
            queue.append([root_left.right, root_right.left])

        return True