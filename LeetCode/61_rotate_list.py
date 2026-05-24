'''
Given the head of a linked list, rotate the list to the right by k places.

 

Example 1:


Input: head = [1,2,3,4,5], k = 2
Output: [4,5,1,2,3]
Example 2:


Input: head = [0,1,2], k = 4
Output: [2,0,1]
 

Constraints:

The number of nodes in the list is in the range [0, 500].
-100 <= Node.val <= 100
0 <= k <= 2 * 109
'''

# solution
from typing import Optional

# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
        
class Solution:
    def rotateRight(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        cnt = 0
        curr = head
        new_head = None

        while curr :
            cnt += 1
            curr = curr.next

        if cnt == 0 :
            return None

        k = k % cnt
        rotate = cnt - k

        if k == 0 :
            return head

        curr = head
        cnt = 0

        while curr :
            cnt += 1
            if cnt == rotate :
                if curr.next :
                    new_head = curr.next
                    curr.next = None

            curr = curr.next
        
        curr = new_head
        
        while curr :
            if not curr.next :
                curr.next = head
                break

            curr = curr.next
        
        return new_head