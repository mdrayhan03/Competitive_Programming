'''

'''

# solution
from typing import Optional

# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def deleteDuplicates(self, head: Optional[ListNode]) -> Optional[ListNode]:
        seen_list = set()
        remove_list = set()
        new_node = None
        curr = head

        while curr :
            if curr.val not in seen_list :
                seen_list.add(curr.val)
            else :
                remove_list.add(curr.val)

            curr = curr.next

        print(seen_list)
        print(remove_list)

        curr = head
        new_head = None

        while curr :
            if curr.val in remove_list :
                curr = curr.next

            else :
                if new_node :
                    new_node.next = ListNode(curr.val)
                    new_node = new_node.next

                else :
                    new_node = ListNode(curr.val)
                    new_head = new_node
            
                curr = curr.next

        return new_head