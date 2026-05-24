'''
Given the head of a linked list, remove the nth node from the end of the list and return its head.

 

Example 1:


Input: head = [1,2,3,4,5], n = 2
Output: [1,2,3,5]
Example 2:

Input: head = [1], n = 1
Output: []
Example 3:

Input: head = [1,2], n = 1
Output: [1]
 

Constraints:

The number of nodes in the list is sz.
1 <= sz <= 30
0 <= Node.val <= 100
1 <= n <= sz
'''

# solution
from typing import Optional

# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
        
class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        cnt = 0
        curr = head

        while curr :
            cnt += 1
            curr = curr.next
        
        remove = cnt + 1 - n
        if remove == 1 :
            head = head.next
            return head

        if cnt == 1 and cnt == remove :
            return None

        cnt = 0
        curr = head

        while curr :
            cnt += 1
            if cnt + 1 == remove :
                if curr.next.next :
                    curr.next = curr.next.next
                else :
                    curr.next = None

                break
            
            curr = curr.next
                
        return head