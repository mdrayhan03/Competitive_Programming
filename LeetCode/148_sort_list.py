'''
Given the head of a linked list, return the list after sorting it in ascending order.

 

Example 1:


Input: head = [4,2,1,3]
Output: [1,2,3,4]
Example 2:


Input: head = [-1,5,3,4,0]
Output: [-1,0,3,4,5]
Example 3:

Input: head = []
Output: []
 

Constraints:

The number of nodes in the list is in the range [0, 5 * 104].
-105 <= Node.val <= 105
'''

# solution
from typing import Optional

# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
        
class Solution:
    def sortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head:
            return None
            
        # Step 1: Collect just the raw primitive values (Very fast)
        vals = []
        curr = head
        while curr:
            vals.append(curr.val)
            curr = curr.next

        # Step 2: Use Python's built-in C-optimized sorting (Incredibly fast)
        vals.sort()

        # Step 3: Put the sorted values back into the existing nodes in-place
        curr = head
        for val in vals:
            curr.val = val
            curr = curr.next
            
        return head