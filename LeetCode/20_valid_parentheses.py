'''
Given a string s containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.

An input string is valid if:

Open brackets must be closed by the same type of brackets.
Open brackets must be closed in the correct order.
Every close bracket has a corresponding open bracket of the same type.
 

Example 1:

Input: s = "()"

Output: true

Example 2:

Input: s = "()[]{}"

Output: true

Example 3:

Input: s = "(]"

Output: false

Example 4:

Input: s = "([])"

Output: true

Example 5:

Input: s = "([)]"

Output: false

 

Constraints:

1 <= s.length <= 104
s consists of parentheses only '()[]{}'.
'''

# solution
import time

class Solution:
    def isValid(self, s: str) -> bool:
        _left = ["(", "{", "["]

        val_list = []
        flag = False

        if len(s) % 2 != 0 :
            return flag

        for ele in s :
            if ele in _left :
                val_list.append(ele)
            else :
                if len(val_list) == 0 :
                    flag = False
                    break
                elif ele == ")" and val_list.pop() == _left[0] :
                    flag = True
                elif ele == "}" and val_list.pop() == _left[1] :
                    flag = True
                elif ele == "]" and val_list.pop() == _left[2] :
                    flag = True
                else :
                    flag = False
                    break

        if len(val_list) != 0 :
            return False
        return flag
        
start_time = time.time()
solution = Solution()
solution.isValid("()")
print(time.time() - start_time)