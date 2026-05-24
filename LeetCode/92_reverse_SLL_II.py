'''
Given the head of a singly linked list and two integers left and right where left <= right, reverse the nodes of the list from position left to position right, and return the reversed list.

 

Example 1:


Input: head = [1,2,3,4,5], left = 2, right = 4
Output: [1,4,3,2,5]
Example 2:

Input: head = [5], left = 1, right = 1
Output: [5]
 

Constraints:

The number of nodes in the list is n.
1 <= n <= 500
-500 <= Node.val <= 500
1 <= left <= right <= n
'''

# solution
from typing import Optional

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseBetween(self, head: Optional[ListNode], left: int, right: int) -> Optional[ListNode]:
        if not head or left == right:
            return head
        
        # Step 1: Create a dummy node to eliminate edge cases where left = 1
        dummy = ListNode(0)
        dummy.next = head
        
        # Step 2: Walk up to the node right BEFORE the sub-segment starts
        left_prev = dummy
        curr = head
        for _ in range(left - 1):
            left_prev = left_prev.next
            curr = curr.next
            
        # At this point, 'curr' is at position 'left', and 'left_prev' is right before it.
        # Bookmark 'curr' because it will become the tail of our reversed sub-segment!
        sub_tail = curr
        
        # Step 3: Perform standard standard in-place reversal for (right - left + 1) nodes
        prev = None
        for _ in range(right - left + 1):
            nxt = curr.next
            curr.next = prev
            prev = curr
            curr = nxt
            
        # Step 4: Stitch the reversed section back into the main list
        left_prev.next = prev  # Connect node before 'left' to the new head of sub-segment
        sub_tail.next = curr   # Connect tail of sub-segment to the node after 'right'
        
        return dummy.next